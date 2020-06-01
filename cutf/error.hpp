#ifndef __CUTF_ERROR_HPP__
#define __CUTF_ERROR_HPP__

#ifndef CUTF_CHECK_ERROR
#define CUTF_CHECK_ERROR(status) cutf::error::check(status, __FILE__, __LINE__, __func__)
#endif
#ifndef CUTF_CHECK_ERROR_M
#define CUTF_CHECK_ERROR_M(status, message) cutf::error::check(status, __FILE__, __LINE__, __func__, message)
#endif

// This macro will be deleted in the future release
#ifndef CUTF_HANDLE_ERROR
#define CUTF_HANDLE_ERROR(status) cutf::error::check(status, __FILE__, __LINE__, __func__)
#endif
#ifndef CUTF_HANDLE_ERROR_M
#define CUTF_HANDLE_ERROR_M(status, message) cutf::error::check(status, __FILE__, __LINE__, __func__, message)
#endif

#endif /* end of include guard */
