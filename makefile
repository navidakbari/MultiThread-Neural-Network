CC=g++

all: Multithread

Multithread: Multithread.o
	$(CC) Multithread.o -lpthread -o Multithread
Multithread.o: Multithread.cpp Multithread.h
	$(CC) -c Multithread.cpp -lpthread

clean:
	rm *.o Multithread