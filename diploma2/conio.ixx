
export module diploma.conio;

import "defs.hpp";

import <Windows.h>;
#undef min
#undef max



const HANDLE console_handle = GetStdHandle(STD_OUTPUT_HANDLE);

EXPORT_BEGIN

struct point2hi
{
	int16_t x, y;
};

void set_console_pos(point2hi pos)
{
	COORD coords{ .X = pos.x, .Y = pos.y };
	SetConsoleCursorPosition(console_handle, coords);
}

point2hi get_console_pos()
{
	CONSOLE_SCREEN_BUFFER_INFO cbsi;
	GetConsoleScreenBufferInfo(console_handle, &cbsi);
	
	return { .x = cbsi.dwCursorPosition.X, .y = cbsi.dwCursorPosition.Y };
}

class cursor_pos_holder
{
	point2hi pos;
public:
	cursor_pos_holder() : pos(get_console_pos()) {}
	~cursor_pos_holder() { set_console_pos(pos); }
};

void clear_line()
{
	CONSOLE_SCREEN_BUFFER_INFO cbsi;
	GetConsoleScreenBufferInfo(console_handle, &cbsi);
	
	cursor_pos_holder _;
	std::print("{1:{0}}", cbsi.dwSize.X, "");
}

EXPORT_END
