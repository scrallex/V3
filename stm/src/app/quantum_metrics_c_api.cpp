extern "C" {

void sep_free_result(const char* result) { delete[] result; }

} // extern "C"