import socket
import json

from detect import detect
def server():
    host=socket.gethostname()
    port=3000
    server_socket=socket.socket()
    server_socket.bind((host,port))
    server_socket.listen(2)
    conn,address=server_socket.accept()
    data=conn.recv(1024).decode()
    print("from connected user: " + str(data))
    detector=detect()
    image=detector.downloadImage(str(data))
    print("image downloaded")
    detector_result=detector.detectImage(image)
    print("image detected")
    prediction=detector.mapToLabel(detector_result)
    print("image labelled")
    data = json.dumps(prediction)
    print(data)
    conn.send(bytes(data.encode('utf-8')))    
    conn.close()
    #only works if tensorflow object detection api is working (uncomment the next line to get the labelled image)
    detector.visualise(detector_result,image)

if __name__ == '__main__':
    server()