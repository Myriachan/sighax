CC = g++ -m64
LD = g++ -m64
CFLAGS = -O2
CXXFLAGS = $(CFLAGS) -std=gnu++11
LDFLAGS = -lgmp

OBJS = sighax.o

sighax: $(OBJS)
	$(LD) -o $@ $^ $(CFLAGS) $(LDFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS) $(CXXFLAGS)
