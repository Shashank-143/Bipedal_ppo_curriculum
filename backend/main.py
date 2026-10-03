import argparse
import uvicorn
from training.train import train

def main():
    parser = argparse.ArgumentParser(description="Backend Application")
    commands = parser.add_subparsers(dest="command", help="Available commands")

    train_cmd = commands.add_parser("train", help="Train the model")
    train_cmd.add_argument("--total-steps", type=int, default=1_000_000)
    train_cmd.add_argument("--learning-rate", type=float, default=3e-4)

    server_cmd = commands.add_parser("serve", help="Run the API server")
    server_cmd.add_argument("--port", type=str, default=8000)

    args = parser.parse_args()

    if args.command == "train":
        train(args.total_steps, learning_rate=args.learning_rate)
    elif args.command == "serve":
        uvicorn.run("api.server:app", host="127.0.0.1", port=int(args.port))

if __name__ == "__main__":
    main()