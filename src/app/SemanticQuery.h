#pragma once

#include <QHash>
#include <QRegularExpression>
#include <QStringList>

struct SemanticQuery
{
    QString canonical;
    QStringList description_words;
};

inline QStringList semanticQueryWords(const QString& text)
{
    return text.toLower().split(QRegularExpression("[^\\p{L}\\p{N}]+"),
                                QString::SkipEmptyParts);
}

// Prefer the longest class/alias phrase; remaining words filter descriptions.
inline SemanticQuery resolveSemanticQuery(
    const QString& text, const QStringList& classes,
    const QHash<QString, QString>& aliases)
{
    const QStringList words = semanticQueryWords(text);
    SemanticQuery result;
    int best_length = 0;
    auto consider = [&](const QString& phrase, const QString& canonical) {
        const QStringList key = semanticQueryWords(phrase);
        if (key.isEmpty() || key.size() <= best_length) return;
        for (int i = 0; i + key.size() <= words.size(); ++i) {
            if (words.mid(i, key.size()) != key) continue;
            best_length = key.size();
            result.canonical = canonical;
            result.description_words = words.mid(0, i) + words.mid(i + key.size());
            return;
        }
    };
    for (const QString& name : classes) {
        consider(name, name);
        consider(name + 's', name);
    }
    QStringList keys = aliases.keys();
    keys.sort();
    for (const QString& key : keys)
        if (classes.contains(aliases.value(key), Qt::CaseInsensitive))
            consider(key, aliases.value(key));
    const QStringList grammar = {"a", "an", "the", "with", "and"};
    for (const QString& word : grammar) result.description_words.removeAll(word);
    return result;
}

inline bool semanticDescriptionMatches(const QString& description,
                                      const QStringList& required_words)
{
    const QStringList words = semanticQueryWords(description);
    for (const QString& word : required_words)
        if (!words.contains(word)) return false;
    return true;
}
