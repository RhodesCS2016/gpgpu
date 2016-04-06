#ifndef MYLIB
#define MYLIB

#include <stdio.h>
#include <stdlib.h>

// use single byte to store cell state in order to increase memory efficiency.
typedef unsigned char ubyte;

// struct to store world parameters
typedef struct World {
  size_t worldWidth;
  size_t worldHeight;
  size_t dataLength;  // worldWidth * worldHeight
} world;

// struct to ease the passing of data to operations
typedef struct Board {
  world *_world;
  ubyte *data;
  ubyte *resultData;
} board;

void initWorld(size_t width, size_t height, world *w);
void initBoard(FILE *fp, board *b, world *w);
void printBoard(ubyte *data, world *w);
void reportTime(size_t iterations, size_t board_size, size_t threads, float time);

#endif
