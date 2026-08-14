import os
import json
import time
import threading

import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

load_dotenv()

API_KEY = os.getenv("ANGEL_API_KEY")
CLIENT_CODE = os.getenv("ANGEL_CLIENT_CODE")
PIN = os.getenv("ANGEL_PIN")
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")


class AngelMarketData:

    def __init__(self):
        self.obj = None
        self.sws = None
        self.data = {}
        self.connected = False

    def login(self):
        self.obj = SmartConnect(api_key=API_KEY)

        totp = pyotp.TOTP(TOTP_SECRET).now()

        session = self.obj.generateSession(
            CLIENT_CODE,
            PIN,
            totp
        )

        if not session.get("status"):
            raise RuntimeError(
                f"Angel One login failed: {session}"
            )

        jwt = session["data"]["jwtToken"]
        feed_token = self.obj.getfeedToken()

        self.sws = SmartWebSocketV2(
            jwt,
            API_KEY,
            CLIENT_CODE,
            feed_token
        )

    def subscribe(self, tokens):
        """
        tokens = list of Angel NSE token strings
        """

        token_list = [
            {
                "exchangeType": 1,
                "tokens": [str(t) for t in tokens]
            }
        ]

        def on_data(wsapp, message):
            token = str(message.get("token"))

            self.data[token] = message

        def on_open(wsapp):
            self.connected = True

            self.sws.subscribe(
                "NSE-SCREENER",
                3,
                token_list
            )

        def on_error(wsapp, error):
            print("Angel WebSocket error:", error)

        def on_close(wsapp):
            self.connected = False

        self.sws.on_data = on_data
        self.sws.on_open = on_open
        self.sws.on_error = on_error
        self.sws.on_close = on_close

        thread = threading.Thread(
            target=self.sws.connect,
            daemon=True
        )

        thread.start()

        time.sleep(2)

    def get(self, token):
        return self.data.get(str(token))

    def close(self):
        if self.sws:
            try:
                self.sws.close_connection()
            except Exception:
                pass