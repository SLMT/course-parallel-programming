#include "gui.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <X11/Xlib.h>

namespace pp {
namespace hw2 {
namespace nbody {

GUI::GUI(unsigned width, unsigned height) {
	// Set the width and the height
	width_ = width;
	height_ = height;

	// Open a connection to the x-window server
	display_ = XOpenDisplay(NULL);
	if(display_ == NULL) {
		fprintf(stderr, "cannot open display\n");
		exit(1);
	}

	// Get the id of the default screen
	screen_id_ = DefaultScreen(display_);

	// Generate the color pixels we need
	color_black_ = BlackPixel(display_, screen_id_);
	color_white_ = WhitePixel(display_, screen_id_);

	// Create a window
	window_ = XCreateSimpleWindow(display_, RootWindow(display_, screen_id_), 0, 0, width_, height_, 0, color_black_, color_white_);

	// Create a graphic context
	XGCValues values;
	gc_ = XCreateGC(display_, window_, 0, &values);
	//XSetForeground(display, gc, color_black_);
	//XSetBackground(display, gc, 0X0000FF00);
	//XSetLineAttributes(display, gc, 1, LineSolid, CapRound, JoinRound);

	// map(show) the window
	XMapWindow(display_, window_);
	XSync(display_, 0);
}

GUI::~GUI() {
}

void GUI::CleanAll() {
	// Draw a rectangle to clean everything
	XSetForeground(display_, gc_, color_black_);
	XFillRectangle(display_, window_, gc_, 0, 0, width_, height_);
}

void GUI::DrawAPoint(unsigned x, unsigned y) {
	XSetForeground(display_, gc_, color_white_);
	XDrawPoint(display_, window_, gc_, x, y);
}

void GUI::Flush() {
	XFlush(display_);
}

} // namespace nbody
} // namespace hw2
} // namespace pp


const double PI = 3.14159;

using pp::hw2::nbody::GUI;

int main(int argc,char *argv[])
{
	GUI *gui = new GUI(500, 500);

	gui->CleanAll();
	for( int i = 0; i < 100; i++ )
		gui->DrawAPoint(int(200*cos(i/100.0*2*PI))+250, 250+int(200*sin(i/100.0*2*PI)));
	for( int i = 0; i < 50; i++)
		gui->DrawAPoint(int(100*cos(i/50.0*2*PI))+250, 250+int(100*sin(i/50.0*2*PI)));
	gui->Flush();

	sleep(10);
	return 0;
}
