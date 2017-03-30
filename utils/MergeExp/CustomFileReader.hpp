/*
 * CustomFileReader.hpp
 *
 *  Created on: Mar 25, 2016
 *      Author: yongchao
 */

#ifndef INCLUDE_CUSTOMFILEREADER_HPP_
#define INCLUDE_CUSTOMFILEREADER_HPP_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

#ifdef COMPRESSED_INPUT
#include <zlib.h>
/*file type*/
typedef gzFile FileReaderDescriptor;

/*open the file*/
#define myfopen(fileName, mode)		gzopen(fileName, mode)

/*open the stdin*/
#define myopenstdin(mode)	gzdopen(fileno(stdin), mode)

/*close the file*/
#define myfclose(file) gzclose(file)

/*check end of file*/
#define myfeof(file) gzeof(file)

/*read data from file*/
#define myfread(buffer, size, nmemb, file) gzread(file, buffer, (size) * (nmemb))

#else	/*COMPRESSED_INPUT*/
/*file type*/
typedef FILE* FileReaderDescriptor;

/*open the file*/
#define myfopen(fileName, mode) fopen(fileName, mode)

/*open the stdin*/
#define myopenstdin(mode) 		stdin

/*close the file*/
#define myfclose(file) fclose(file)

/*check the end-of-file*/
#define myfeof(file) feof(file)

/*read data from file*/
#define myfread(buffer, size, nmemb, file) fread(buffer, size, nmemb, file)

#endif	/*COMPRESSED_INPUT*/

/*define the structure*/
class CustomFileReader {
public:
	CustomFileReader() {
		_fp = NULL;

		/*create the file buffer*/
		_fileBufferR = new char[4096 + 8];
		_fileBufferSentinel = 0;
		_fileBufferLength = 0;
		if (_fileBufferR == NULL) {
			fprintf(stderr, "Memory allocation failed in file %s in line %d\n",
					__FUNCTION__, __LINE__);
			exit(-1);
		}
		_fileBuffer = _fileBufferR + 8; /*make it aligned*/
	}
	~CustomFileReader() {
		close();
		if (_fileBufferR) {
			delete[] _fileBufferR;
		}
	}
	inline ssize_t getline(char**buffer, size_t* bufferSize) {
		int32_t ch;
		size_t length = 0;

		/*get the character*/
		while ((ch = getchar()) != -1 && ch != '\n') {
			/*check the \r*/
			if (ch == '\r') {
				/*get the next character*/
				if ((ch = getchar()) != '\n') {
					ungetchar(ch);
				}
				break;
			}
			/*resize the buffer*/
			if (*buffer == NULL || length + 1 >= *bufferSize) {
				*bufferSize = *buffer != NULL ? *bufferSize * 2 : (1 << 11);
				*buffer = (char*) realloc(*buffer, *bufferSize);
				if (!*buffer) {
					fprintf(stderr, "Memory reallocation failed (%ld)\n",
							*bufferSize * 2);
					exit(-1);
				}
			}
			(*buffer)[length++] = ch;
		}

		/*end of file and no character is read in*/
		if (ch == -1 && length == 0) {
			return -1;
		}
		/*test the memory buffer*/
		if (*buffer == NULL) {
			*bufferSize = length + 1;
			*buffer = (char*) malloc(*bufferSize * sizeof(char));
			if (!*buffer) {
				fprintf(stderr, "Memory reallocation failed (%ld)\n",
						*bufferSize * 2);
				exit(-1);
			}
		}
		/*set the end of line*/
		(*buffer)[length] = '\0';

		/*remove the '\r' character*/
		for (;
				length > 0
						&& ((*buffer)[length - 1] == '\r'
								|| (*buffer)[length - 1] == '\n'); --length) {
			(*buffer)[length - 1] = '\0';
		}

		return length;
	}
	/*open the file*/
	inline FileReaderDescriptor open(const char* fileName, const char* mode) {
		/*close the file*/
		if (_fp) {
			myfclose(_fp);
		}
		if (!fileName) {
			_fp = myopenstdin(mode);
			if (!_fp) {
				fprintf(stderr, "Failed to open STDIN\n");
				exit(-1);
			}
		} else {
			_fp = myfopen(fileName, mode);
			if (!_fp) {
				fprintf(stderr, "Failed to open file: %s\n", fileName);
				exit(-1);
			}
		}
		return _fp;
	}

	/*close the file*/
	inline void close() {
		if (_fp) {
			myfclose(_fp);
		}
		_fp = NULL;
	}

	inline int32_t getchar() {
		/*check the end-of-file*/
		if (_fileBufferSentinel >= _fileBufferLength) {
			/*re-fill the buffer*/
			_fileBufferSentinel = 0;
			/*read file*/
			_fileBufferLength = myfread(_fileBuffer, 1, 4096, _fp);
			if (_fileBufferLength == 0) {
				/*reach the end of the file*/
				if (myfeof(_fp)) {
					return -1;
				} else {
					fprintf(stderr, "File reading failed in function %s line %d\n",
							__FUNCTION__, __LINE__);
					exit(-1);
				}
			}
		}
		/*return the current character, and increase the sentinel position*/
		return _fileBuffer[_fileBufferSentinel++];
	}
	inline int32_t ungetchar(int32_t ch) {
		if (_fileBufferSentinel >= 0) {
			_fileBuffer[--_fileBufferSentinel] = ch;
		} else {
			fprintf(stderr, "Two consecutive ungetc operations occurred\n");
			return -1; /*an error occurred, return end-of-file marker*/
		}
		return ch;
	}

	/*file pointer*/
	FileReaderDescriptor _fp;
	char* _fileBufferR;
	char* _fileBuffer;
	int32_t _fileBufferLength;
	int32_t _fileBufferSentinel;
};


#endif /* INCLUDE_CUSTOMFILEREADER_HPP_ */
