import logging
from typing import Dict, Any, List, Optional

from config import config
from agent.graph import define_graph
from agent.state import empty_working_memory
from utils.callback_handler import AgentTraceCallback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeAgent:
    def __init__(self):
        config.validate_chat_config()
        self.app = define_graph()

    def chat(self, user_input: str) -> str:
        """
        Main entry point for the agent.
        Each call is treated as an independent research session.
        """
        inputs = {
            "input": user_input,
            "plan": [],
            "current_step": 0,
            "working_memory": empty_working_memory(),
            "messages": [],
            "executor_call_count": 0,
            "iteration_count": 0,
        }

        tracer: Optional[AgentTraceCallback] = None
        run_config: Dict[str, Any] = {"recursion_limit": 150}

        if config.trace_enabled:
            tracer = AgentTraceCallback(
                trace_dir=config.trace_dir,
                stream_to_console=config.trace_console,
                color=config.trace_color,
            )
            run_config["callbacks"] = [tracer]

        try:
            final_state = self.app.invoke(inputs, config=run_config)
            response = final_state.get("response", "I could not generate a response.")
            return response

        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            return f"An error occurred: {str(e)}"

        finally:
            if tracer:
                tracer.close()

    def reset(self):
        pass
