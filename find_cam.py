import socket
import subprocess
import os
import re

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def scan_rtsp_port():
    local_ip = get_local_ip()
    print("=" * 60)
    print(f"Your PC's IP Address: {local_ip}")
    print("=" * 60)
    
    if local_ip == '127.0.0.1':
        print("Error: No active network connection found!")
        return
        
    subnet_prefix = '.'.join(local_ip.split('.')[:3])
    print(f"Scanning subnet {subnet_prefix}.1 to {subnet_prefix}.254 on RTSP Port 554...\n")
    
    # Priority IPs to check first
    priority_ips = [
        f"{subnet_prefix}.120",
        f"{subnet_prefix}.49",
        f"{subnet_prefix}.100",
        f"{subnet_prefix}.108",
        f"{subnet_prefix}.64",
        f"{subnet_prefix}.200",
        f"{subnet_prefix}.10",
        f"{subnet_prefix}.2"
    ]
    
    # Quick scan all in subnet
    all_ips = list(dict.fromkeys(priority_ips + [f"{subnet_prefix}.{i}" for i in range(1, 255)]))
    
    found_cams = []
    for ip in all_ips:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.15)
        res = sock.connect_ex((ip, 554))
        sock.close()
        if res == 0:
            print(f"[FOUND CAMERA] IP Camera detected at: {ip} (Port 554 OPEN)")
            found_cams.append(ip)
            
    print("-" * 60)
    if found_cams:
        print(f"Found {len(found_cams)} active camera(s): {', '.join(found_cams)}")
        print("\nTry running:")
        for cam_ip in found_cams:
            print(f'python qr_detector.py "rtsp://admin:Smarden%4012@{cam_ip}:554/video/live?channel=1&subtype=0&unicast=true&proto=Onvif"')
    else:
        print("No devices responded on Port 554 in subnet " + subnet_prefix + ".x")
        print("\nPlease check:")
        print(" 1. Is your PC connected to the exact same Wi-Fi / Router as the camera?")
        print(" 2. Is the camera powered ON with Ethernet / Wi-Fi connected?")
    print("=" * 60)

if __name__ == "__main__":
    scan_rtsp_port()
