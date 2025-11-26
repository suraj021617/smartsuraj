from app import app

with app.test_client() as client:
    response = client.get('/power-simple')
    print(f'Status Code: {response.status_code}')
    if response.status_code == 200:
        print('SUCCESS! Route works!')
        print(f'Response length: {len(response.data)} bytes')
    else:
        print('FAILED! Route returned 404')
        print(f'Response: {response.data.decode()[:200]}')
