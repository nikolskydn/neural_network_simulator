/** @addtogroup Data
 * @{*/

/** @file */

#ifndef DATACREATOR_GASPARYANMOSESINFORMIKA
#define DATACREATOR_GASPARYANMOSESINFORMIKA

#include "data.hpp"
#include "datapcnni2003.hpp"
#include "dataunn270117.hpp"

#include <memory>

namespace NNSimulator{

template<typename T>
inline std::unique_ptr<Data<T>> Data<T>::createItem( ChildId id ){
	std::unique_ptr<Data<T>> ptr;
	switch( id )
	{
		case DataPCNNI2003Id:
			ptr = std::unique_ptr<Data<T>>( std::make_unique<DataPCNNI2003<T>>() );
		break;
		case DataUNN270117Id:
			ptr = std::unique_ptr<Data<T>>( std::make_unique<DataUNN270117<T>>() );
		break;
	}
	return ptr;
}

}

#endif // DATACREATOR_GASPARYANMOSESINFORMIKA

/*@}*/
