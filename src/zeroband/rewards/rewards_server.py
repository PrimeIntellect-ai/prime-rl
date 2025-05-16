from fastapi import FastAPI, Request, Response
from requests import get
import json
import argparse

from zeroband.inference.rewards import compute_rewards, RewardRequest, RewardsResponse

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Prime Rewards API Server")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
parser.add_argument("--auth", type=str, required=True, help="Authentication password")
args = parser.parse_args()

PORT = args.port
AUTH = args.auth

try:
    ip_addr = get('https://api.ipify.org').content.decode('utf8')
    print(f"IP Address: {ip_addr}")
    print(f"Port: {PORT}")
    print(f"To connect to the server, use the following URL: http://{ip_addr}:{PORT}/compute_rewards")
except Exception as e:
    print(f"Could not determine IP address: {e}")


app = FastAPI(title="Prime Rewards API")

@app.post("/compute_rewards")
async def compute_rewards_endpoint(request: Request):
    if request.headers.get("Authorization") != f"Bearer {AUTH}":
        return Response(content="Unauthorized", status_code=401)

    try:
        body = await request.body()
        reward_request: RewardRequest = RewardRequest.model_validate(json.loads(body))
        reward_response: RewardsResponse = compute_rewards(reward_request)
        reward_json = reward_response.model_dump_json()

        return Response(
            content=reward_json,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=compute_rewards.json"}
        )

    except Exception as e:
        return Response(
            content=f"Error processing json: {str(e)}",
            status_code=400
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)