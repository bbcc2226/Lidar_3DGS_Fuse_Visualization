#include "NavigationIntentParser.h"

#include <QRegularExpression>
#include <QStringList>

namespace
{
struct NavigationPhrase
{
    const char* text;
};

// Longest phrases come first so a specific polite form wins before a shorter
// command can consume it.
constexpr NavigationPhrase kNavigationPhrases[] = {
    {"please bring me to"},
    {"please take me to"},
    {"could you bring me to"},
    {"could you take me to"},
    {"bring me to"},
    {"take me to"},
    {"navigate to"},
    {"please go to"},
    {"go to"},
    {"navigate"},
};

bool matchesOneOf(const QString& command, const QStringList& alternatives)
{
    return alternatives.contains(command);
}
} // namespace

QString NavigationIntentParser::normalize(const QString& input)
{
    QString normalized = input.toLower().trimmed();
    normalized.replace(QRegularExpression("[\\t\\r\\n]+"), " ");
    normalized.replace(QRegularExpression("\\s+"), " ");
    normalized.remove(QRegularExpression("[.!?,;:]+$"));
    return normalized.trimmed();
}

ParsedNavigationCommand NavigationIntentParser::parse(
    const QString& input) const
{
    ParsedNavigationCommand result;
    const QString command = normalize(input);
    if (command.isEmpty()) {
        result.error = "Please enter a navigation command.";
        return result;
    }

    if (matchesOneOf(command, {"stop", "stop navigation", "cancel navigation"})) {
        result.intent = NavigationIntent::Stop;
        return result;
    }
    if (matchesOneOf(command, {"pause", "pause navigation"})) {
        result.intent = NavigationIntent::Pause;
        return result;
    }
    if (matchesOneOf(command,
                     {"resume", "resume navigation", "continue", "continue navigation"})) {
        result.intent = NavigationIntent::Resume;
        return result;
    }

    for (const NavigationPhrase& phrase : kNavigationPhrases) {
        const QString prefix = QString::fromLatin1(phrase.text);
        if (command == prefix) {
            result.intent = NavigationIntent::NavigateTo;
            result.error = "Please provide a destination.";
            return result;
        }
        if (!command.startsWith(prefix + ' ')) continue;

        result.intent = NavigationIntent::NavigateTo;
        result.destination = command.mid(prefix.size()).trimmed();
        // Articles are grammatical rather than part of the destination key.
        if (result.destination.startsWith("the "))
            result.destination.remove(0, 4);
        static const QRegularExpression relation_re(
            "^(.+?)\\s+(next\\s+to|near|on|left\\s+of|right\\s+of)\\s+(?:the\\s+)?(.+)$");
        const QRegularExpressionMatch relation_match = relation_re.match(result.destination);
        if (relation_match.hasMatch()) {
            result.destination = relation_match.captured(1).trimmed();
            result.relation = relation_match.captured(2).simplified();
            result.reference = relation_match.captured(3).trimmed();
            if (result.destination.startsWith("the ")) result.destination.remove(0, 4);
        }
        if (result.destination.isEmpty())
            result.error = "Please provide a destination.";
        return result;
    }

    result.error = "I could not recognize that navigation command.";
    return result;
}
