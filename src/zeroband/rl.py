from libtmux import Server
from libtmux.constants import PaneDirection
from pydantic_config import BaseConfig, parse_argv
# from zeroband.train import Config as TrainConfig
# from zeroband.infer import Config as InferConfig


class RLConfig(BaseConfig):

    # train: TrainConfig
    # infer: InferConfig

    kill_sessions: bool = False


_TRAINING_SESSION_NAME = 'training'
_INFERENCE_SESSION_NAME = 'inference'


def main(config: RLConfig):
    # Connect to the tmux server
    server = Server()

    # Handle existing sessions
    for session_name in [_TRAINING_SESSION_NAME, _INFERENCE_SESSION_NAME]:
        if server.has_session(session_name):
            old_session = server.sessions.get(session_name=session_name)
            if config.kill_sessions:
                old_session.kill()
                print(f"Killed existing session '{session_name}'")
            else:
                raise ValueError(f"Session '{session_name}' already exists and kill_sessions is False")

    # Create training session
    training_session = server.new_session(session_name=_TRAINING_SESSION_NAME)
    training_window = training_session.active_window
    training_window.rename_window("monitor")

    # Setup left pane with top
    left_pane = training_window.active_pane
    left_pane.send_keys('top')

    # Split window and setup right pane with nvtop
    # Use PaneDirection.Right instead of "vertical"
    right_pane = training_window.split(direction=PaneDirection.Right)
    right_pane.send_keys('nvtop')

    # Create inference session
    inference_session = server.new_session(session_name=_INFERENCE_SESSION_NAME)
    inference_window = inference_session.active_window
    inference_window.rename_window("inference")
    inference_pane = inference_window.active_pane
    inference_pane.send_keys('echo "Inference session initialized"')

    print(f"Sessions '{_TRAINING_SESSION_NAME}' and '{_INFERENCE_SESSION_NAME}' created successfully")
    print("You can attach to them using: tmux attach-session -t training")
    print("Or: tmux attach-session -t inference")

if __name__ == "__main__":
    main(RLConfig(**parse_argv()))