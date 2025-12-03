import time
import json
from google.cloud import pubsub_v1
from google.api_core.exceptions import DeadlineExceeded
from ..service.inference_services import  set_global_models, process_inference_task
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
        payload = json.loads(data)

        # hapus field yang tidak diperlukan
        for meta_field in ["project_id_env", "topic_id_env"]:
            payload.pop(meta_field, None)


        subscriber.acknowledge(
            request={
                "subscription": subscription_path,
                "ack_ids": [msg.ack_id],
            }
        )
        print("🟢 Message acknowledged.\n")
        print("🧾 Clean payload:", payload)
        process_inference_task(payload)