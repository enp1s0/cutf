#ifndef __CUTF_ARITHMETIC_HPP__
#define __CUTF_ARITHMETIC_HPP__
#include "type.hpp"
#include "macro.hpp"

namespace cutf {
namespace arithmetic {

template <class T>
CUTF_DEVICE_HOST_FUNC T add(const T a, const T b) {return a + b;}
template <> CUTF_DEVICE_HOST_FUNC cuComplex       add<cuComplex      >(const cuComplex a, const cuComplex b) {return make_cuComplex(a.x + b.x, a.y + b.y);}
template <> CUTF_DEVICE_HOST_FUNC cuDoubleComplex add<cuDoubleComplex>(const cuDoubleComplex a, const cuDoubleComplex b) {return make_cuDoubleComplex(a.x + b.x, a.y + b.y);}

template <class T>
CUTF_DEVICE_HOST_FUNC T sub(const T a, const T b) {return a - b;}
template <> CUTF_DEVICE_HOST_FUNC cuComplex       sub<cuComplex      >(const cuComplex a      , const cuComplex b      ) {return make_cuComplex      (a.x - b.x, a.y - b.y);}
template <> CUTF_DEVICE_HOST_FUNC cuDoubleComplex sub<cuDoubleComplex>(const cuDoubleComplex a, const cuDoubleComplex b) {return make_cuDoubleComplex(a.x - b.x, a.y - b.y);}

template <class T>
CUTF_DEVICE_HOST_FUNC T mul(const T a, const T b) {return a * b;}
template <> CUTF_DEVICE_HOST_FUNC cuComplex       mul<cuComplex      >(const cuComplex a      , const cuComplex       b) {return make_cuComplex      (a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);}
template <> CUTF_DEVICE_HOST_FUNC cuDoubleComplex mul<cuDoubleComplex>(const cuDoubleComplex a, const cuDoubleComplex b) {return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);}

template <class T>
CUTF_DEVICE_HOST_FUNC T mad(const T a, const T b, const T c) {return add(mul(a, b), c);}

template <class T>
CUTF_DEVICE_HOST_FUNC typename cutf::type::real_type<T>::type abs2(const T a) {return a * a;};
template <> CUTF_DEVICE_HOST_FUNC float  abs2(const cuComplex       a) {return a.x * a.x + a.y * a.y;};
template <> CUTF_DEVICE_HOST_FUNC double abs2(const cuDoubleComplex a) {return a.x * a.x + a.y * a.y;};
} // namespace complex
} // namespace cutf
#endif // __CUTF_COMPLEX_HPP__
