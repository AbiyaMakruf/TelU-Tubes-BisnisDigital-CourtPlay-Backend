import time
from google.cloud import pubsub_v1
from google.api_core.exceptions import DeadlineExceeded

def process_ai_task(message_data: str):
    print(f"🚀 Starting AI inference on message: {message_data}")
    time.sleep(5)
    print(f"✅ Finished processing message: {message_data}")

def pull_and_process(project_id: str, subscription_id: str):
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)
    print(f"📡 Listening for messages on {subscription_path} ...")

    while True:
        try:
            response = subscriber.pull(
                request={
                    "subscription": subscription_path,
                    "max_messages": 1,
                },
                timeout=10,
            )
        except DeadlineExceeded:
            print("⏳ Timeout — no new messages, retrying...")
            time.sleep(2)
            continue

        if not response.received_messages:
            print("⚪ No messages yet, waiting...")
            time.sleep(2)
            continue

        msg = response.received_messages[0]
        data = msg.message.data.decode("utf-8")

        process_ai_task(data)

        subscriber.acknowledge(
            request={
                "subscription": subscription_path,
                "ack_ids": [msg.ack_id],
            }
        )
        print("🟢 Message acknowledged.\n")

if __name__ == "__main__":
    pull_and_process(
        project_id="courtplay-analytics-474615",
        subscription_id="inference-pull"
    )
