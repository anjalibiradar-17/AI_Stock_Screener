import os
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect

load_dotenv()

api_key = os.getenv("ANGEL_API_KEY")
client_code = os.getenv("ANGEL_CLIENT_CODE")
pin = os.getenv("ANGEL_PIN")
totp_secret = os.getenv("ANGEL_TOTP_SECRET")

obj = SmartConnect(api_key=api_key)

totp = pyotp.TOTP(totp_secret).now()

session = obj.generateSession(client_code, pin, totp)

if session.get("status"):
    print("ANGEL ONE LOGIN: SUCCESS")
    print("Client:", client_code)
    print("Feed token: GENERATED")
else:
    print("ANGEL ONE LOGIN: FAILED")
    print(session)