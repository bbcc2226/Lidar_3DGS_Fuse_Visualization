#include "../src/app/SemanticQuery.h"
#include <gtest/gtest.h>

TEST(SemanticQuery, PreservesClassesAliasesAndPlurals)
{
    const QStringList classes = {"mug", "microwave", "coffee table"};
    const QHash<QString, QString> aliases = {{"micro", "microwave"}};
    EXPECT_EQ(resolveSemanticQuery("micro", classes, aliases).canonical, "microwave");
    const auto q = resolveSemanticQuery("mugs", classes, aliases);
    EXPECT_EQ(q.canonical, "mug");
    EXPECT_TRUE(q.description_words.isEmpty());
    EXPECT_TRUE(resolveSemanticQuery("unknown", classes, aliases).canonical.isEmpty());
}

TEST(SemanticQuery, FiltersWholeWordsIgnoringCaseAndPunctuation)
{
    const auto q = resolveSemanticQuery("the WHITE mug with a blue handle", {"mug"}, {});
    EXPECT_EQ(q.canonical, "mug");
    EXPECT_TRUE(semanticDescriptionMatches("A white mug; blue-handle.", q.description_words));
    EXPECT_FALSE(semanticDescriptionMatches("A white mug with a red handle", q.description_words));
    EXPECT_FALSE(semanticDescriptionMatches("A white mug with a blueberry handle", q.description_words));
    EXPECT_FALSE(semanticDescriptionMatches("", q.description_words));
}

TEST(SemanticQuery, ChoosesLongestClassAndAlias)
{
    const auto q = resolveSemanticQuery("black coffee table", {"table", "coffee table"}, {});
    EXPECT_EQ(q.canonical, "coffee table");
    EXPECT_EQ(q.description_words, QStringList{"black"});
    const auto alias = resolveSemanticQuery("silver micro oven", {"microwave"},
                                           {{"micro oven", "microwave"}});
    EXPECT_EQ(alias.canonical, "microwave");
    EXPECT_EQ(alias.description_words, QStringList{"silver"});
}
