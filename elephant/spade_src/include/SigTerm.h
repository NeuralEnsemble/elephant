/* 
 *  File: SigTerm.h
 *  Copyright (c) 2020 Florian Porrmann
 *  
 *  MIT License
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *  
 */

// Based on sigint.c from Christian Borgelt
#pragma once

#include <signal.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX // Disable the build in MIN/MAX macros to prevent collisions
#endif
#include <windows.h>
#else
#define _POSIX_C_SOURCE 200809L
#endif

#ifdef WITH_SIG_TERM
static volatile sig_atomic_t aborted = 0;
#ifndef _WIN32
static struct sigaction sigOld;
static struct sigaction sigNew;
#endif

void sigAbort(const int& state)
{
	aborted = state;
}

#ifdef _WIN32

static BOOL WINAPI sigHandler(DWORD type)
{
	if (type == CTRL_C_EVENT || type == CTRL_CLOSE_EVENT || type == CTRL_LOGOFF_EVENT || type == CTRL_SHUTDOWN_EVENT)
		sigAbort(-1);
	return TRUE;
}

void sigInstall()
{
	SetConsoleCtrlHandler(sigHandler, TRUE);
}

void sigRemove()
{
	SetConsoleCtrlHandler(sigHandler, FALSE);
}

#else

static void sigHandler(int type)
{
	if (type == SIGINT)
		sigAbort(-1);
}

void sigInstall()
{
	sigNew.sa_handler = sigHandler;
	sigNew.sa_flags   = 0;
	sigemptyset(&sigNew.sa_mask);
	sigaction(SIGINT, &sigNew, &sigOld);
}

void sigRemove()
{
	sigaction(SIGINT, &sigOld, reinterpret_cast<struct sigaction*>(0));
}
#endif

int sigAborted()
{
	return aborted;
}
#endif
