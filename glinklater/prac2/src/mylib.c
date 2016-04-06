#include "mylib.h"

void initWorld(size_t width, size_t height, world *w) {
  w->worldWidth = width;
  w->worldHeight = height;
  w->dataLength = width*height;
}

void initBoard(FILE *fp, board *b, world *w) {
  ubyte *data = (ubyte*)malloc(w->dataLength);
  ubyte *resultData = (ubyte*)malloc(w->dataLength);

  int count;
  char ch = '\0';
  while (count < w->dataLength) {
    ch = (char)fgetc(fp);
    // printf("%c, %d\n", ch, count);
    if (ch != '\0' || ch != ',' || ch != '\n') {
      if (ch == '1') {
        data[count++] = 1;
      } else if (ch == '0') {
        data[count++] = 0;
      }
    }
  }
  b->_world = w;
  b->data = data;
  b->resultData = resultData;
}

void printBoard(ubyte *data, world *w) {
  for (int count = 0; count < w->dataLength; count++) {
    if (count % w->worldWidth == 0) printf("\n");
    ubyte i = data[count];
    if (i == 1) printf("x");
    else if (i == 0) printf(" ");
    else printf("%d", i);
  }
  printf("\n---\n");
}

void reportTime(size_t iterations, size_t board_size, size_t threads, float milli) {
  printf("%zu,%zu,%zu,%f\n", board_size, iterations, threads, milli);
}
