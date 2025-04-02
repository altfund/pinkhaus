mkdir -p ./letta
# If LETTA_SERVER_PASSWORD isn't set, the server will autogenerate a password
 docker run --name letta -v ./pgdata:/var/lib/postgresql/data -p 8283:8283 --env-file .env letta/letta:latest
