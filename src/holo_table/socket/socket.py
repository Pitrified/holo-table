"""Simple UPD socket wrapper for sending and receiving messages.

A server binds to a port and listens for incoming messages.
A client connects to a server and sends messages.
"""
import socket
from time import sleep
from typing import Self
import socket
from typing import Any, Generator, Self, TypeAlias
from loguru import logger as lg

# from socket import _RetAddress
_RetAddress: TypeAlias = Any


class UdpSocketSender:
    """A simple UDP socket sender."""

    def __init__(self, ip: str, port: int) -> None:
        """Initialize the socket."""
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, message: str) -> None:
        """Send a message to the server."""
        lg.debug(f"Sending: {message}")
        message_bytes = message.encode("utf-8")
        self.sock.sendto(message_bytes, (self.ip, self.port))

    def quit(self) -> None:
        """Send a quit message to the server."""
        self.send("quit")

    def __enter__(self) -> Self:
        """Return the socket object when used as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the socket when used as a context manager."""
        self.quit()


class UdpSocketReceiver:
    """A simple UDP socket receiver."""

    def __init__(self, ip: str, port: int) -> None:
        """Initialize the socket."""
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))

    def receive(
        self,
        buffer_size: int,
    ) -> Generator[tuple[str, _RetAddress], None, None]:
        """Receive messages from the client.

        Stop receiving when the client sends a "quit" message.

        Should be iterated over in a loop.

        Args:
            buffer_size (int): The size of the buffer to receive the message, in bytes.
        """
        while True:
            data, addr = self.receive_once(buffer_size)
            if data == "quit":
                return
            yield data, addr

    def receive_once(self, buffer_size: int) -> tuple[str, _RetAddress]:
        """Receive a single message from the client.

        Args:
            buffer_size (int): The size of the buffer to receive the message, in bytes.

        Returns:
            tuple[str, _RetAddress]: The message and the address of the client.
        """
        data_bytes, addr = self.sock.recvfrom(buffer_size)
        data = data_bytes.decode("utf-8")
        return data, addr

    def close(self) -> None:
        """Close the socket."""
        lg.debug("Closing socket")
        self.sock.close()

    def __enter__(self) -> Self:
        """Return the socket object when used as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the socket when used as a context manager."""
        self.close()
