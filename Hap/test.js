const ws = new WebSocket('ws://10.22.64.107:8000');
ws.onopen = () => console.log('Connected to WebSocket!');
ws.onerror = (err) => console.error('WebSocket Error:', err);
