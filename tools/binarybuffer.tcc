/*
 * binarybuffer.tcc
 *
 * this file contains only realization
 * of methods in BinaryBuffers.
 *
 * this file must be included only in
 * binarybuffer.h
 *
 */
#include <cstring>
#include <string>

template<typename T>
inline BinaryBufferOutS& BinaryBufferOutS::operator<< ( const T& t){
    ostr_.write(reinterpret_cast<const char*>(&t), sizeof(T));

	#if DEBUGBINARYBUFFER >= 1
    std::cout << "element " << t << " has been written" << std::endl;
	#endif

    return *this;
}

template<>
inline BinaryBufferOutS& BinaryBufferOutS::operator<<( const std::string& s ){
	if ( !s.empty() )
		ostr_.write( s.c_str(), s.size() );		// + 1 ?

	#if DEBUGBINARYBUFFER >= 1
	std::cout << "std::string \"" << s << "\" has been written" << std::endl;
	#endif

	return *this;
}

inline BinaryBufferOutS& BinaryBufferOutS::operator<<( const char* t ){
    ostr_.write(t, std::strlen(t));		// + 1 ?

	#if DEBUGBINARYBUFFER >= 1
    std::cout << "c string \"" << t << "\" has been written" << std::endl;
	#endif

    return *this;
}

template<typename T>
inline BinaryBufferOutS& BinaryBufferOutS::operator<<(const std::vector<T>& vec){
	if ( ! vec.empty() )
		ostr_.write( reinterpret_cast<const char*>(vec.data()), vec.size()*sizeof(T) );

	#if DEBUGBINARYBUFFER >= 1
	std::cout << "vector [";
	for(size_t i=0; i < vec.size(); ++i){
		std::cout << " " << vec[i];
	}
	std::cout << "] has been written" << std::endl;
	#endif

	return *this;
}

template<typename T>
inline BinaryBufferOutS& BinaryBufferOutS::operator<<(const std::valarray<T>& val){
	if ( val.size() > 0 )
		ostr_.write( reinterpret_cast<const char*>(&val[0]), val.size()*sizeof(T) );

//	for(size_t i=0; i < val.size(); ++i){
//		ostr_.write( reinterpret_cast<const char*>(&val[i]), sizeof(T) );
//	}

	#if DEBUGBINARYBUFFER >= 1
	std::cout << "valarray [";
	for(size_t i=0; i < val.size(); ++i){
		std::cout << " " << val[i];
	}
	std::cout << "] has been written" << std::endl;
	#endif

	return *this;
}

// -------------------------------- BinaryBufferIn

template<typename T>
inline BinaryBufferInS& BinaryBufferInS::operator>>(T& elem){
	bf_.read(reinterpret_cast<char*>(&elem), sizeof(T));

	#if DEBUGBINARYBUFFER >= 1
	std::cout << "element " << elem << " has been read" << std::endl;
	#endif

	return *this;
}

template<>
inline BinaryBufferInS& BinaryBufferInS::operator>> (std::string& s){
	if ( ! s.empty() )
		bf_.read( reinterpret_cast<char*>(&s[0]), s.size() );

	#if DEBUGBINARYBUFFER >= 1
	std::cout << "std::string \"" << s << "\" has been read" << std::endl;
	#endif

	return *this;
}

template<typename T>
inline BinaryBufferInS& BinaryBufferInS::operator>> (std::vector<T>& vec){
	if ( ! vec.empty() )
		bf_.read( reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(T) );

	#if DEBUGBINARYBUFFER >= 1
	std::cout << "vector [";
	for(size_t i=0; i < vec.size(); ++i){
		std::cout << " " << vec[i];
	}
	std::cout << "] has been read" << std::endl;
	#endif

	return *this;
}

template<typename T>
inline BinaryBufferInS& BinaryBufferInS::operator>> (std::valarray<T>& val){
	if ( val.size() > 0 )
		bf_.read( reinterpret_cast<char*>(&val[0]), val.size() * sizeof(T) );

	#if DEBUGBINARYBUFFER >= 1
	std::cout << "valarray [";
	for(size_t i=0; i < val.size(); ++i){
		std::cout << " " << val[i];
	}
	std::cout << "] has been read" << std::endl;
	#endif

	return *this;
}

inline BinaryBufferInS& BinaryBufferInS::operator>> (char* s){
	bf_.read( s, std::strlen(s) );

	#if DEBUGBINARYBUFFER >= 1
	std::cout << "c string \"" << s << "\" has been read" << std::endl;
	#endif

	return *this;
}
