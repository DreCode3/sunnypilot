#!/bin/bash
# Restore MTU to 1500 (default)
# Run with: sudo ./set_mtu_1500.sh

IFACE=$(route get 192.168.8.236 2>/dev/null | awk '/interface:/{print $2}')

if [ -z "$IFACE" ]; then
    echo "Could not find interface for 192.168.8.236 — is VPN connected?"
    exit 1
fi

echo "Restoring MTU to 1500 on $IFACE"
sudo ifconfig "$IFACE" mtu 1500
echo "Done. Current MTU:"
ifconfig "$IFACE" | grep mtu
