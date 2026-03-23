#ifndef ZNZLIB_H
#define ZNZLIB_H

#include <stdio.h>
#include <zlib.h>

typedef struct znzFile_s {
    int withz;
    FILE* nzfptr;
    gzFile zfptr;
} *znzFile;

znzFile znzopen(const char* path, const char* mode, int use_compression);
int     znzclose(znzFile f);
size_t  znzread(void* buf, size_t size, size_t nmemb, znzFile f);
size_t  znzwrite(const void* buf, size_t size, size_t nmemb, znzFile f);
long    znztell(znzFile f);
int     znzseek(znzFile f, long offset, int whence);
int     znzeof(znzFile f);
int     znz_isnull(znzFile f);

#endif
