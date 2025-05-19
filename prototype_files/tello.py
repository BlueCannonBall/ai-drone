import socket
import crcmod
from bitstring import BitStream

class Tello:
    def __init__(self):
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.conn.connect(("192.168.10.1", 8890))
        self.crc8_func = crcmod.mkCrcFun(0x11D, initCrc=0x00, rev=True, xorOut=0x00)
        self.crc16_func = crcmod.mkCrcFun(0x11021, initCrc=0x0000, rev=True, xorOut=0x0000)
        self.send_seq_number = 0
    
    def send(self, type, message_id, payload):
        seq_number = self.send_seq_number

        while True:
            buf = BitStream()
            buf.append("0xCC") # Header
            buf.append(('uint', 13, len(payload))) # Payload length
            buf.append(('uint', 8, self.crc8_func(buf.tobytes()))) # Hash of header and payload length
            buf.append("bin=01") # Packet is drone-bound
            buf.append(('uint', 3, type)) # Packet type
            buf.append("bin=000") # Packet subtype (always zero)
            buf.append(('uint', 16, message_id))
            buf.append(('uint', 16, seq_number)) # Sequence number
            # buf.append(())
        
        self.send_seq_number += 1

