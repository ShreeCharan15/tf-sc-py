
import socket


def client():
    host = socket.gethostname()
    port = 3000 
    client_socket = socket.socket()
    client_socket.connect((host, port)) 
    message = input(" -> ") 
    client_socket.send(message.encode()) 
    data = client_socket.recv(1024).decode() 
    print('Received from server: ' + data)
    client_socket.close() 

if __name__ == '__main__':
    client()