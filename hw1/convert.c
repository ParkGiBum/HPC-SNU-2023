#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void fallback_print_usage() {
  printf("Usage: ./convert [int|long|float|double] number\n");
  printf("Example: ./convert float 3.14\n");
  exit(0);
}

void print_int(int x) {
  // one character per one bit
  char output[32 + 1] = {
      0,
  };


  for (int i = 31; i >= 0; ) {
      if ((x >> i) & 1) {
          output[31 - i] = '1';
      } else {
          output[31 - i] = '0';
      }
      i = i - 1;
  }

  printf("%s\n", output);
}

void print_long(long x) {
  // one character per one bit
  char output[64 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  for (int i = 63; i >= 0; i--) {
      if ((x >> i) & 1) {
          output[63 - i] = '1';
      } else {
          output[63 - i] = '0';
      }
  }

  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_float(float x) {
  // one character per one bit
  char output[32 + 1] = {
      0,
  };
  /* YOUR CODE START HERE */

  unsigned int* zero = (unsigned int*)&x;
  for (int i = 31; i >= 0; i--) {
      if ((*zero >> i) & 1) {
          output[31 - i] = '1';
      } else {
          output[31 - i] = '0';
      }
  }


  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_double(double x) {
  // one character per one bit
  char output[64 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  unsigned long long* zero = (unsigned long long*)&x;
  for (int i = 63; i >= 0; i--) {
      if ((*zero >> i) & 1) {
          output[63 - i] = '1';
      } else {
          output[63 - i] = '0';
      }
  }

  /* YOUR CODE END HERE */
  printf("%s\n", output);
}

int main(int argc, char **argv) {
  if (argc != 3)
    fallback_print_usage();
  if (strcmp(argv[1], "int") == 0) {
    print_int(atoi(argv[2]));
  } else if (strcmp(argv[1], "long") == 0) {
    print_long(atol(argv[2]));
  } else if (strcmp(argv[1], "float") == 0) {
    print_float(atof(argv[2]));
  } else if (strcmp(argv[1], "double") == 0) {
    print_double(atof(argv[2]));
  } else {
    fallback_print_usage();
  }
}
