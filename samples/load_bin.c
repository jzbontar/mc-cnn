#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int main(void)
{
	int fd;

	float *left, *right, *disp;

	fd = open("../left.bin", O_RDONLY);
	left = mmap(NULL, 1 * 70 * 370 * 1226 * sizeof(float), PROT_READ, MAP_SHARED, fd, 0);
	close(fd);

	fd = open("../right.bin", O_RDONLY);
	right = mmap(NULL, 1 * 70 * 370 * 1226 * sizeof(float), PROT_READ, MAP_SHARED, fd, 0);
	close(fd);

	fd = open("../disp.bin", O_RDONLY);
	disp = mmap(NULL, 1 * 1 * 370 * 1226 * sizeof(float), PROT_READ, MAP_SHARED, fd, 0);
	close(fd);

	return 0;
}
