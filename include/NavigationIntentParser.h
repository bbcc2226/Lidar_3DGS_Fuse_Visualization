#pragma once

#include <QString>

enum class NavigationIntent
{
    NavigateTo,
    Stop,
    Pause,
    Resume,
    Unknown
};

struct ParsedNavigationCommand
{
    NavigationIntent intent = NavigationIntent::Unknown;
    QString destination;
    QString relation;
    QString reference;
    QString error;

    bool isValid() const
    {
        return intent != NavigationIntent::Unknown && error.isEmpty();
    }
};

class NavigationIntentParser
{
public:
    ParsedNavigationCommand parse(const QString& input) const;

private:
    static QString normalize(const QString& input);
};
