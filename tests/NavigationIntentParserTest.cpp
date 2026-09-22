#include "NavigationIntentParser.h"

#include <gtest/gtest.h>

TEST(NavigationIntentParserTest, ExtractsDestinationsFromSupportedPhrases)
{
    const NavigationIntentParser parser;
    const std::vector<std::pair<QString, QString>> examples = {
        {"go to the kitchen", "kitchen"},
        {"Navigate to sofa", "sofa"},
        {"navigate front door", "front door"},
        {"please bring me to the coffee table", "coffee table"},
        {"Please take me to Bedroom!", "bedroom"},
        {"could you bring me to the entrance?", "entrance"},
    };

    for (const auto& example : examples) {
        const ParsedNavigationCommand result = parser.parse(example.first);
        EXPECT_TRUE(result.isValid()) << example.first.toStdString();
        EXPECT_EQ(result.intent, NavigationIntent::NavigateTo);
        EXPECT_EQ(result.destination, example.second);
    }
}

TEST(NavigationIntentParserTest, ReplacesSpatialRelationOnNextRequest)
{
    NavigationIntentParser parser;

    const ParsedNavigationCommand first =
        parser.parse("navigate to the mug near the water filter");
    EXPECT_EQ(first.destination, "mug");
    EXPECT_EQ(first.relation, "near");
    EXPECT_EQ(first.reference, "water filter");

    const ParsedNavigationCommand second =
        parser.parse("navigate to the mug on the shelf");
    EXPECT_EQ(second.destination, "mug");
    EXPECT_EQ(second.relation, "on");
    EXPECT_EQ(second.reference, "shelf");
}

TEST(NavigationIntentParserTest, RecognizesPlaybackControlIntents)
{
    const NavigationIntentParser parser;
    EXPECT_EQ(parser.parse("stop navigation").intent, NavigationIntent::Stop);
    EXPECT_EQ(parser.parse("Pause.").intent, NavigationIntent::Pause);
    EXPECT_EQ(parser.parse("continue navigation").intent, NavigationIntent::Resume);
    EXPECT_TRUE(parser.parse("resume").isValid());
}

TEST(NavigationIntentParserTest, ReportsMissingDestination)
{
    const NavigationIntentParser parser;
    for (const QString& command :
         {QString("go to"), QString("navigate"), QString("please bring me to")}) {
        const ParsedNavigationCommand result = parser.parse(command);
        EXPECT_EQ(result.intent, NavigationIntent::NavigateTo);
        EXPECT_FALSE(result.isValid());
        EXPECT_EQ(result.error, "Please provide a destination.");
    }
}

TEST(NavigationIntentParserTest, RejectsUnknownOrEmptyCommands)
{
    const NavigationIntentParser parser;
    EXPECT_EQ(parser.parse("show me the map").intent, NavigationIntent::Unknown);
    EXPECT_FALSE(parser.parse("show me the map").error.isEmpty());
    EXPECT_EQ(parser.parse("   ").intent, NavigationIntent::Unknown);
    EXPECT_FALSE(parser.parse("   ").isValid());
}

TEST(NavigationIntentParserTest, RequiresPrefixWordBoundary)
{
    const NavigationIntentParser parser;
    const ParsedNavigationCommand result = parser.parse("navigation settings");
    EXPECT_EQ(result.intent, NavigationIntent::Unknown);
    EXPECT_TRUE(result.destination.isEmpty());
}
