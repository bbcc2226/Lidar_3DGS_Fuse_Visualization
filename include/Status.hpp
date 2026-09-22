#pragma once

// Shared error contract. Check ok() before reading Result<T>::value(), and
// add stage context when propagating failures across component boundaries.

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace lio_visual_ba
{

enum class ErrorCode
{
    kOk = 0,
    kInvalidArgument,
    kNotFound,
    kIoError,
    kParseError,
    kFailedPrecondition,
    kDataLoss,
    kNumericalFailure,
    kUnimplemented,
    kInternal,
};

constexpr std::string_view ErrorCodeName(ErrorCode code) noexcept
{
    switch (code)
    {
    case ErrorCode::kOk:
        return "OK";
    case ErrorCode::kInvalidArgument:
        return "INVALID_ARGUMENT";
    case ErrorCode::kNotFound:
        return "NOT_FOUND";
    case ErrorCode::kIoError:
        return "IO_ERROR";
    case ErrorCode::kParseError:
        return "PARSE_ERROR";
    case ErrorCode::kFailedPrecondition:
        return "FAILED_PRECONDITION";
    case ErrorCode::kDataLoss:
        return "DATA_LOSS";
    case ErrorCode::kNumericalFailure:
        return "NUMERICAL_FAILURE";
    case ErrorCode::kUnimplemented:
        return "UNIMPLEMENTED";
    case ErrorCode::kInternal:
        return "INTERNAL";
    }
    return "UNKNOWN";
}

class Status
{
  public:
    Status() = default;

    static Status Ok() noexcept
    {
        return Status();
    }

    static Status Error(ErrorCode code, std::string message)
    {
        if (code == ErrorCode::kOk)
        {
            return Status(ErrorCode::kInternal, "an error Status cannot use ErrorCode::kOk");
        }
        return Status(code, std::move(message));
    }

    bool ok() const noexcept
    {
        return code_ == ErrorCode::kOk;
    }
    explicit operator bool() const noexcept
    {
        return ok();
    }
    ErrorCode code() const noexcept
    {
        return code_;
    }
    const std::string& message() const noexcept
    {
        return message_;
    }

    std::string ToString() const
    {
        if (ok())
        {
            return std::string(ErrorCodeName(code_));
        }
        return std::string(ErrorCodeName(code_)) + ": " + message_;
    }

    // Add high-level context while retaining the original error category.
    Status WithContext(std::string_view context) const
    {
        if (ok() || context.empty())
        {
            return *this;
        }
        if (message_.empty())
        {
            return Status(code_, std::string(context));
        }
        return Status(code_, std::string(context) + ": " + message_);
    }

  private:
    Status(ErrorCode code, std::string message) : code_(code), message_(std::move(message)) {}

    ErrorCode code_ = ErrorCode::kOk;
    std::string message_;
};

template <typename T> class Result
{
  public:
    static Result Success(T value)
    {
        return Result(std::move(value));
    }

    static Result Failure(Status status)
    {
        if (status.ok())
        {
            status =
                Status::Error(ErrorCode::kInternal, "a failed Result cannot contain an OK Status");
        }
        return Result(std::move(status));
    }

    bool ok() const noexcept
    {
        return std::holds_alternative<T>(storage_);
    }

    explicit operator bool() const noexcept
    {
        return ok();
    }

    const Status& status() const noexcept
    {
        if (const auto* error = std::get_if<Status>(&storage_))
        {
            return *error;
        }
        static const Status ok_status = Status::Ok();
        return ok_status;
    }

    T& value() &
    {
        CheckHasValue();
        return std::get<T>(storage_);
    }

    const T& value() const&
    {
        CheckHasValue();
        return std::get<T>(storage_);
    }

    T&& value() &&
    {
        CheckHasValue();
        return std::get<T>(std::move(storage_));
    }

  private:
    explicit Result(T value) : storage_(std::move(value)) {}
    explicit Result(Status status) : storage_(std::move(status)) {}

    void CheckHasValue() const
    {
        if (!ok())
        {
            throw std::logic_error("accessed failed Result: " + status().ToString());
        }
    }

    std::variant<T, Status> storage_;
};

} // namespace lio_visual_ba
