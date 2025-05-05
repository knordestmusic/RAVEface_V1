from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server

def dump(address, *args):
    print(f"< {address} {args}")

disp = Dispatcher()
disp.set_default_handler(dump)

ip   = "127.0.0.1"
port = 8000         # choose any free port
print(f"Listening on {ip}:{port}")
server = osc_server.ThreadingOSCUDPServer((ip, port), disp)
server.serve_forever()
