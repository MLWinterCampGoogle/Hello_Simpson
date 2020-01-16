from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
from random import randint
from bs4 import BeautifulSoup
from requests_toolbelt.multipart import decoder
from simple_test import predict
import re

def parse_multipart(content, boundary):
    boundary = bytes('--' + boundary, 'utf-8')
    content = content[content.find(b'\r\n\r\n')+4:]
    content = content[:content.find(boundary)]
    return content

class WebServer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.png') or self.path.endswith('.jpg') or self.path.endswith('.img'):
            f = open(self.path[1:], 'rb')
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            self.wfile.write(f.read())
            f.close()
        else:
            self.path = '/index.html'
            try:
                file_to_open = open(self.path[1:]).read()
                self.send_response(200)
            except Exception as e:
                print(e)
                file_to_open = 'File not found'
                self.send_response(404)

            self.end_headers()
            self.wfile.write(bytes(file_to_open, 'utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        content_type = self.headers['Content-Type']
        img_bytes = self.rfile.read(content_length)
        boundary = re.match('.*boundary=(.*)$', content_type).group(1)
        body = parse_multipart(img_bytes, boundary)
        pid = str(randint(0, 0x7FFFFFFF))
        print(pid)
        f = open('photo/{}.img'.format(pid), 'wb')
        f.write(body)
        f.close()
        predict(pid + '.img')
        with open('index.html') as fin:
            soup = BeautifulSoup(fin.read())
        print(self.headers)
        res_img = soup.new_tag('img', src='/result/{}.jpg'.format(pid))
        soup.body.append(res_img)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(bytes(str(soup), 'utf-8'))

httpd = HTTPServer(('0.0.0.0', 80), WebServer)
httpd.serve_forever()
