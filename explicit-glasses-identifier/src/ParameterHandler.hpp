#ifndef PARAMETERHANDLER_HPP
#define PARAMETERHANDLER_HPP

#include <cstring>
#include <string>
#include <sstream>
#include <vector>

/**
 * Splits a string into
 * @param s             the string which should be splitted
 * @param delim     the delim character
 * @param result     the container for the splitted substrings
 */
template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

/**
 * Splits a string in substring by delimiters
 * @param s             the string which should be splitted
 * @param delim     the delimiter for splitting
 * @return the splitted string
 */
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}


/**
 * Return the argument value by its key
 * @param argc          the number of arguments
 * @param argv          the argument values
 * @param param       the key of the argument value which should be extracted
 * @return the argument value of the given key
 */
std::string getInputParameter(int argc, const char** argv, const char* param) {
        
        std::string paramString(param);
        std::vector<std::string> keys = split(paramString, '|');
        
        for (int argIt = 0; argIt < argc - 1; argIt++) {
                for(unsigned int keyIt = 0; keyIt < keys.size(); keyIt++){
                        if (strcmp(argv[argIt], keys[keyIt].c_str()) == 0) {
                                return std::string(argv[argIt + 1]);
                        }
                }                
        }
        return "";
}


/**
 * Checks if a key exists in the arguments
 * @param argc          the number of arguments
 * @param argv          the argument values
 * @param param       the key of the argument value which should be extracted
 * @return true if the given key exists in the argument list or false if not
 */
bool existsParameter(int argc, const char** argv, const char* param) {
        for (int i = 0; i < argc; i++) {
                if (strcmp(argv[i], param) == 0) {
                        return true;
                }
        }
        return false;
}

#endif /* PARAMETERHANDLER_HPP */

