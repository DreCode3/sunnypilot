#!/bin/bash
# Set MTU to 1280 for SSH over mobile VPN to Comma 4
# Run with: sudo ./set_mtu_1280.sh

IFACE=$(route get 192.168.8.236 2>/dev/null | awk '/interface:/{print $2}')

if [ -z "$IFACE" ]; then
    echo "Could not find interface for 192.168.8.236 — is VPN connected?"
    exit 1
fi

echo "Setting MTU to 1280 on $IFACE"
sudo ifconfig "$IFACE" mtu 1280
echo "Done. Current MTU:"
ifconfig "$IFACE" | grep mtu
