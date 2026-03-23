/*
 * znzlib.c
 * gzip-transparent I/O, from nifti_clib.
 */

#include "znzlib.h"
#include <stdlib.h>
#include <string.h>

znzFile znzopen(const char* path, const char* mode, int use_compression)
{
    znzFile f = (znzFile)calloc(1, sizeof(struct znzFile_s));
    if (!f) return NULL;

    f->withz = use_compression;

    if (use_compression) {
        f->zfptr = gzopen(path, mode);
        f->nzfptr = NULL;
        if (!f->zfptr) { free(f); return NULL; }
    } else {
        f->nzfptr = fopen(path, mode);
        f->zfptr = NULL;
        if (!f->nzfptr) { free(f); return NULL; }
    }

    return f;
}

int znzclose(znzFile f)
{
    int ret = 0;
    if (!f) return 0;
    if (f->withz) {
        ret = gzclose(f->zfptr);
    } else {
        ret = fclose(f->nzfptr);
    }
    free(f);
    return ret;
}

size_t znzread(void* buf, size_t size, size_t nmemb, znzFile f)
{
    if (!f) return 0;
    if (f->withz) {
        int bytes = gzread(f->zfptr, buf, (unsigned)(size * nmemb));
        if (bytes < 0) return 0;
        return (size_t)bytes / size;
    } else {
        return fread(buf, size, nmemb, f->nzfptr);
    }
}

size_t znzwrite(const void* buf, size_t size, size_t nmemb, znzFile f)
{
    if (!f) return 0;
    if (f->withz) {
        int bytes = gzwrite(f->zfptr, buf, (unsigned)(size * nmemb));
        if (bytes < 0) return 0;
        return (size_t)bytes / size;
    } else {
        return fwrite(buf, size, nmemb, f->nzfptr);
    }
}

long znztell(znzFile f)
{
    if (!f) return -1;
    if (f->withz) return (long)gztell(f->zfptr);
    return ftell(f->nzfptr);
}

int znzseek(znzFile f, long offset, int whence)
{
    if (!f) return -1;
    if (f->withz) return gzseek(f->zfptr, offset, whence) >= 0 ? 0 : -1;
    return fseek(f->nzfptr, offset, whence);
}

int znzeof(znzFile f)
{
    if (!f) return 1;
    if (f->withz) return gzeof(f->zfptr);
    return feof(f->nzfptr);
}

int znz_isnull(znzFile f)
{
    if (!f) return 1;
    if (f->withz) return (f->zfptr == NULL);
    return (f->nzfptr == NULL);
}
