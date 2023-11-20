# import json
# import redis

# def test():
#     redis_uri = 'rediss://default:AVNS_p5SxXC8sjRJE8JkNqB9@task-queue-minhnhatdo0709-a715.a.aivencloud.com:17468'
#     redis_client = redis.from_url(redis_uri, decode_responses=True)
#     task = redis_client.rpop('taskQueue')
#     task = json.loads(task)
#     print(f"task: {vars(task)}")
#     return "Hello World!"

# if __name__ == '__main__':
#     test()

# import requests

# r = requests.get("http://127.0.0.1:1227/ping")
# print(f"content {r.content.decode('utf-8')}")