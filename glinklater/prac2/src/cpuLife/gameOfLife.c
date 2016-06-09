#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <utility>
#include <time.h>
#include "../mylib.h"

board *gameBoard;
world *gameWorld;
FILE *out_file;
clock_t start;
float milli;

/*
 * countAliveCells
 * Evaluate the number of cells in the Moore Neighbourhood that are alive.
 */
inline ubyte countAliveCells(
    size_t x0,
    size_t x1,
    size_t x2,
    size_t y0,
    size_t y1,
    size_t y2
  ) {
  return gameBoard->data[x0 + y0] + gameBoard->data[x1 + y0]
    + gameBoard->data[x2 + y0] + gameBoard->data[x0 + y1]
    + gameBoard->data[x2 + y1] + gameBoard->data[x0 + y2]
    + gameBoard->data[x1 + y2] + gameBoard->data[x2 + y2];
}

/*
 * computeIterationSerial
 * For each cell in the data buffer evaluate if the cell should live or die
 * according to the rules and write to results buffer.
 */
void computeIterationSerial(ubyte *data, world *gameWorld, ubyte *resultData) {
  for (size_t y = 0; y < gameWorld->worldHeight; ++y) {
    // Compute cell coordinates from 1D space to 2D space.
    size_t y0 = ((y + gameWorld->worldHeight - 1) % gameWorld->dataLength);
    size_t y1 = y * gameWorld->worldWidth;
    size_t y2 = ((y + 1) % gameWorld->dataLength);

    for (size_t x = 0; x < gameWorld->worldWidth; ++x) {
      size_t x0 = (x + gameWorld->worldWidth - 1) % gameWorld->worldWidth;
      size_t x2 = (x + 1) % gameWorld->worldWidth;

      ubyte aliveCells = countAliveCells(x0, x, x2, y0, y1, y2);
      // Write evaluated cell state.
      resultData[y1 + x] =
        aliveCells == 3 || (aliveCells == 2 && gameBoard->data[x + y1]) ? 1 : 0;
    }
  }
  // Switch result buffer and data buffer for next iteration.
  std::swap(gameBoard->data, gameBoard->resultData);
}

int main(int argc, char **argv) {
  int iterations = 10000;
  char *in_filename = NULL;
  char *out_filename = NULL;
  size_t board_size = 48;
  gameWorld = (world*)malloc(sizeof(world));
  gameBoard = (board*)malloc(sizeof(board));

  opterr = 0;
  int c;

  while ((c = getopt (argc, argv, (const char*)"o:f:i:s:")) != -1) {
    switch (c) {
      case 'i':
        iterations = atoi(optarg);
        break;
      case 'f':
        in_filename = optarg;
        break;
      case 'o':
        out_filename = optarg;
        break;
      case 's':
        board_size = atoi(optarg);
        break;
      case '?':
        break;
      default:
        break;
    }
  }

  // printf("iterations: %d\n", iterations);
  // printf("in_file: %s\n", in_filename);
  // printf("out_file: %s\n", out_filename);
  // printf("\n");

  if (!in_filename) {
    printf("Please specify a board file\n");
  }

  initWorld(board_size, board_size, gameWorld);
  initBoard(fopen(in_filename, "r"), gameBoard, gameWorld);
  if (out_filename) out_file = fopen(out_filename, "w+");

  start = clock();
  for (int i = 0; i<=iterations; i++) {
    // printBoard(gameBoard->data, gameWorld);
    computeIterationSerial(gameBoard->data, gameBoard->_world, gameBoard->resultData);
  }
  milli = (float)(clock() - start);
  // printBoard(gameBoard->data, gameWorld);
  reportTime(iterations, board_size, 1, milli);
  fprintf(stderr, "cpuLife successfully executed.\n");

  free(gameBoard->data);
  free(gameBoard->resultData);
  if (out_filename) fclose(out_file);
  // printf("\n");
}
