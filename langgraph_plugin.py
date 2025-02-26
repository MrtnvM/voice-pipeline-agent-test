from __future__ import annotations

import base64
import inspect
import json
import os
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Dict,
    List,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
)

import httpx
from langchain_core.messages import AIMessage
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    llm,
    utils,
)
from livekit.agents.llm import LLMCapabilities, ToolChoice
from livekit.agents.llm.function_context import (
    _create_ai_function_info,
    _is_optional_type,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

# LangGraph imports
from langgraph.graph import Graph
from langgraph.graph.message import AnyMessage
import logging

logger = logging.getLogger('langgraph.plugin')


@dataclass
class LLMOptions:
    user: str | None
    temperature: float | None
    parallel_tool_calls: bool | None
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] | None
    caching: Literal["ephemeral"] | None = None


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        graph: Graph,
        api_key: str | None = None,  # Kept for compatibility
        base_url: str | None = None,  # Kept for compatibility
        user: str | None = None,
        client: Any | None = None,  # Kept for compatibility
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
        caching: Literal["ephemeral"] | None = None,
    ) -> None:
        """
        Create a new instance of LangGraph LLM.

        graph (Graph): The LangGraph graph to use.
        api_key (str | None): Not used with LangGraph. Kept for compatibility.
        base_url (str | None): Not used with LangGraph. Kept for compatibility.
        user (str | None): The user identifier. Defaults to None.
        client (Any | None): Not used with LangGraph. Kept for compatibility.
        temperature (float | None): The temperature parameter. Defaults to None.
        parallel_tool_calls (bool | None): Whether to parallelize tool calls. Defaults to None.
        tool_choice (Union[ToolChoice, Literal["auto", "required", "none"]] | None): The tool choice. Defaults to "auto".
        caching (Literal["ephemeral"] | None): Caching configuration. Defaults to None.
        """

        super().__init__(
            capabilities=LLMCapabilities(
                requires_persistent_functions=True,
                supports_choices_on_int=True,
            )
        )

        if graph is None:
            raise ValueError("LangGraph graph is required")

        self._graph = graph
        self._opts = LLMOptions(
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            caching=caching,
        )

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]]
        | None = None,
    ) -> "LLMStream":
        if temperature is None:
            temperature = self._opts.temperature
        if parallel_tool_calls is None:
            parallel_tool_calls = self._opts.parallel_tool_calls
        if tool_choice is None:
            tool_choice = self._opts.tool_choice

        # Convert LiveKit chat context to LangGraph input format
        langgraph_input = self._prepare_langgraph_input(
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

        # Get stream from LangGraph
        stream = self._graph.astream(langgraph_input)

        return LLMStream(
            self,
            langgraph_stream=stream,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options,
        )

    def _prepare_langgraph_input(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] | None = None,
    ) -> Dict[str, Any]:
        """
        Prepare the input for the LangGraph.
        """
        # Extract messages
        messages = self._convert_messages(chat_ctx.messages)
        
        # Build config
        config = {
            "temperature": temperature,
        }
        
        if parallel_tool_calls is not None:
            config["parallel_tool_calls"] = parallel_tool_calls
            
        # Extract tools if available
        tools = None
        if fnc_ctx and len(fnc_ctx.ai_functions) > 0:
            tools = self._convert_functions(fnc_ctx)
            
            # Handle tool choice
            if tool_choice is not None:
                if isinstance(tool_choice, ToolChoice) and tool_choice.type == "function":
                    config["tool_choice"] = {"type": "tool", "name": tool_choice.name}
                elif isinstance(tool_choice, str):
                    if tool_choice == "required":
                        config["tool_choice"] = {"type": "any"}
                    elif tool_choice == "none":
                        config["tool_choice"] = {"type": "none"}
                    else:  # auto
                        config["tool_choice"] = {"type": "auto"}
        
        # Prepare the final input structure
        input_data = {
            "messages": messages,
            "config": config,
        }
        
        if tools:
            input_data["tools"] = tools
            
        if self._opts.user:
            input_data["user"] = self._opts.user
            
        return input_data

    def _convert_messages(self, messages: List[llm.ChatMessage]) -> List[Dict[str, Any]]:
        """
        Convert LiveKit chat messages to LangGraph format.
        """
        result = []
        
        for msg in messages:
            if msg.role not in ["system", "user", "assistant", "tool"]:
                continue
                
            converted_msg = {"role": msg.role}
            
            # Handle content
            if isinstance(msg.content, str):
                converted_msg["content"] = msg.content
            elif isinstance(msg.content, list):
                converted_msg["content"] = []
                for item in msg.content:
                    if isinstance(item, str):
                        converted_msg["content"].append({"type": "text", "text": item})
                    elif isinstance(item, llm.ChatImage):
                        converted_msg["content"].append(self._convert_image(item, id(self)))
            
            # Handle tool calls for assistant messages
            if msg.tool_calls and msg.role == "assistant":
                converted_msg["tool_calls"] = []
                for call in msg.tool_calls:
                    converted_msg["tool_calls"].append({
                        "id": call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call.function_info.name,
                            "arguments": call.arguments
                        }
                    })
            
            # Handle tool responses
            if msg.role == "tool" and msg.tool_call_id:
                # For tool messages, attach tool_call_id
                converted_msg["tool_call_id"] = msg.tool_call_id
                # Content should be formatted correctly
                if isinstance(msg.content, dict):
                    converted_msg["content"] = json.dumps(msg.content)
                
                # Mark as error if there was an exception
                if msg.tool_exception:
                    converted_msg["is_error"] = True
            
            result.append(converted_msg)
            
        return result

    def _convert_functions(self, fnc_ctx: llm.FunctionContext) -> List[Dict[str, Any]]:
        """
        Convert LiveKit functions to LangGraph tool format.
        """
        tools = []
        
        for fnc in fnc_ctx.ai_functions.values():
            # Build parameters schema
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            for arg_name, arg_info in fnc.arguments.items():
                param_schema = self._build_parameter_schema(arg_info)
                parameters["properties"][arg_name] = param_schema
                
                # Add to required list if no default
                if arg_info.default is inspect.Parameter.empty:
                    parameters["required"].append(arg_name)
            
            # Create the tool definition
            tool = {
                "type": "function",
                "function": {
                    "name": fnc.name,
                    "description": fnc.description,
                    "parameters": parameters
                }
            }
            
            tools.append(tool)
            
        return tools

    def _build_parameter_schema(self, arg_info: llm.function_context.FunctionArgInfo) -> Dict[str, Any]:
        """
        Build JSON Schema for function parameter.
        """
        def type_to_str(t: type) -> str:
            if t is str:
                return "string"
            elif t is int:
                return "integer"
            elif t is float:
                return "number"
            elif t is bool:
                return "boolean"
            raise ValueError(f"Unsupported type {t} for parameter")

        # Get the actual type, handling Optional
        is_optional, inner_type = _is_optional_type(arg_info.type)
        
        schema: Dict[str, Any] = {}
        
        # Handle description
        if arg_info.description:
            schema["description"] = arg_info.description
            
        # Handle array types
        if get_origin(inner_type) is list:
            item_type = get_args(inner_type)[0]
            schema["type"] = "array"
            schema["items"] = {"type": type_to_str(item_type)}
            
            # Add enum if choices are available
            if arg_info.choices:
                schema["items"]["enum"] = arg_info.choices
        else:
            # Handle scalar types
            schema["type"] = type_to_str(inner_type)
            
            # Add enum if choices are available
            if arg_info.choices:
                schema["enum"] = arg_info.choices
                
        return schema

    def _convert_image(self, image: llm.ChatImage, cache_key: Any) -> Dict[str, Any]:
        """
        Convert a LiveKit ChatImage to LangGraph format.
        """
        if isinstance(image.image, str):  # image is a URL
            if not image.image.startswith("data:"):
                raise ValueError("LiveKit LangGraph Plugin: Image URLs must be data URLs")

            try:
                header, b64_data = image.image.split(",", 1)
                media_type = header.split(";")[0].split(":")[1]

                supported_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
                if media_type not in supported_types:
                    raise ValueError(
                        f"LiveKit LangGraph Plugin: Unsupported media type {media_type}. Must be jpeg, png, webp, or gif"
                    )

                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "data": b64_data,
                        "media_type": cast(
                            Literal["image/jpeg", "image/png", "image/gif", "image/webp"],
                            media_type,
                        ),
                    }
                }
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"LiveKit LangGraph Plugin: Invalid image data URL {str(e)}"
                )
        elif isinstance(image.image, rtc.VideoFrame):  # image is a VideoFrame
            if cache_key not in image._cache:
                # Use internal implementation to avoid re-encoding each time
                opts = utils.images.EncodeOptions()
                if image.inference_width and image.inference_height:
                    opts.resize_options = utils.images.ResizeOptions(
                        width=image.inference_width,
                        height=image.inference_height,
                        strategy="scale_aspect_fit",
                    )

                encoded_data = utils.images.encode(image.image, opts)
                image._cache[cache_key] = base64.b64encode(encoded_data).decode("utf-8")

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "data": image._cache[cache_key],
                    "media_type": "image/jpeg",
                }
            }

        raise ValueError(
            "LiveKit LangGraph Plugin: ChatImage must be an rtc.VideoFrame or a data URL"
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        langgraph_stream: Awaitable[Any],
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, conn_options=conn_options
        )
        self._awaitable_langgraph_stream = langgraph_stream
        self._langgraph_stream = None

        # Current function call tracking
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_arguments: str | None = None

        self._request_id: str = ""
        self._input_tokens = 0
        self._output_tokens = 0

    async def _run(self) -> None:
        retryable = True
        try:
            if not self._langgraph_stream:
                self._langgraph_stream = self._awaitable_langgraph_stream

            # Process the stream
            async for chunk in self._langgraph_stream:
                chat_chunk = self._parse_langgraph_chunk(chunk)
                if chat_chunk is not None:
                    self._event_ch.send_nowait(chat_chunk)
                    retryable = False

            # Send final usage information
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    request_id=self._request_id,
                    usage=llm.CompletionUsage(
                        completion_tokens=self._output_tokens,
                        prompt_tokens=self._input_tokens,
                        total_tokens=self._input_tokens + self._output_tokens,
                    ),
                )
            )
        except Exception as e:
            # Convert errors to LiveKit error types
            if hasattr(e, "timeout"):
                raise APITimeoutError(retryable=retryable)
            elif hasattr(e, "status_code"):
                raise APIStatusError(
                    str(e),
                    status_code=getattr(e, "status_code", 500),
                    request_id=getattr(e, "request_id", ""),
                    body=getattr(e, "body", ""),
                )
            else:
                raise APIConnectionError(retryable=retryable) from e

    def _parse_langgraph_chunk(self, chunk: dict[str, Any]) -> llm.ChatChunk | None:
        """
        Parse a LangGraph chunk into a LiveKit chat chunk.
        
        The exact implementation depends on the LangGraph output format.
        This is a general structure that needs to be adapted to the specific
        output format of your LangGraph implementation.
        """
        # Generate request ID if not available
        if not self._request_id:
            self._request_id = getattr(chunk, "id", f"lg-{id(chunk)}")
        
        if "chatbot" in chunk:
            chatbot = chunk["chatbot"]
            if "messages" in chatbot:
                messages: list[AnyMessage] = chatbot["messages"]
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        msg: AIMessage = cast(AIMessage, msg)
                        return llm.ChatChunk(
                            request_id=self._request_id,
                            choices=[
                                llm.Choice(
                                    delta=llm.ChoiceDelta(content=msg.content, role="assistant")
                                )
                            ],
                        )
        
        # Handle text content
        if hasattr(chunk, "content") and isinstance(chunk.content, str) and chunk.content:
            return llm.ChatChunk(
                request_id=self._request_id,
                choices=[
                    llm.Choice(
                        delta=llm.ChoiceDelta(content=chunk.content, role="assistant")
                    )
                ],
            )
        
        # Extract token usage if available
        if hasattr(chunk, "usage"):
            usage = getattr(chunk, "usage", {})
            if hasattr(usage, "prompt_tokens"):
                self._input_tokens = usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                self._output_tokens = usage.completion_tokens
                
        # Handle text content
        if hasattr(chunk, "content") and isinstance(chunk.content, str) and chunk.content:
            return llm.ChatChunk(
                request_id=self._request_id,
                choices=[
                    llm.Choice(
                        delta=llm.ChoiceDelta(content=chunk.content, role="assistant")
                    )
                ],
            )
        
        # Handle tool calls
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            # Make sure we have a function context
            if not self._fnc_ctx:
                logger.warning("Received tool call but no function context available")
                return None
                
            tool_calls_info = []
            for tool_call in chunk.tool_calls:
                # Create function info
                tool_id = getattr(tool_call, "id", f"tc-{id(tool_call)}")
                fnc_name = getattr(tool_call.function, "name", "")
                fnc_args = getattr(tool_call.function, "arguments", "{}")
                
                fnc_info = _create_ai_function_info(
                    self._fnc_ctx,
                    tool_id,
                    fnc_name,
                    fnc_args,
                )
                tool_calls_info.append(fnc_info)
                self._function_calls_info.append(fnc_info)

            return llm.ChatChunk(
                request_id=self._request_id,
                choices=[
                    llm.Choice(
                        delta=llm.ChoiceDelta(
                            role="assistant", tool_calls=tool_calls_info
                        ),
                    )
                ],
            )

        # Return None if we couldn't extract anything useful
        return None