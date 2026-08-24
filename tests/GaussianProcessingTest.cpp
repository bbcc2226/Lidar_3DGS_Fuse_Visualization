#include "3dgsProcessing.h"

#include <gtest/gtest.h>

#include <QTemporaryFile>

namespace
{
bool writeText(QTemporaryFile& file, const QByteArray& text)
{
    return file.open() && file.write(text) == text.size() && file.flush();
}
} // namespace

TEST(GaussianProcessingTest, LoadsRawDegreeZeroGaussianParameters)
{
    QTemporaryFile file;
    ASSERT_TRUE(writeText(file,
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\n"
        "property float rot_2\nproperty float rot_3\n"
        "end_header\n"
        "1 2 3 0.1 0.2 0.3 -1.5 -2 -3 -4 1 0 0 0\n"));

    GaussianSplatProcessing processing;
    ASSERT_TRUE(processing.loadPly(file.fileName().toStdString()))
        << processing.lastError();
    ASSERT_EQ(processing.points().size(), 1U);

    const GaussianPoint& point = processing.points().front();
    EXPECT_FLOAT_EQ(point.x, 1.0f);
    EXPECT_FLOAT_EQ(point.scale[1], -3.0f);
    EXPECT_FLOAT_EQ(point.rotation[0], 1.0f);
    EXPECT_FLOAT_EQ(point.opacity, -1.5f);
    EXPECT_FLOAT_EQ(point.sh_dc[2], 0.3f);
    EXPECT_TRUE(processing.metadata().isComplete3DGS());
    EXPECT_EQ(processing.metadata().sh_degree, 0);
}

TEST(GaussianProcessingTest, PreservesDegreeOneSphericalHarmonicCoefficients)
{
    QTemporaryFile file;
    ASSERT_TRUE(writeText(file,
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float f_rest_0\nproperty float f_rest_1\n"
        "property float f_rest_2\nproperty float f_rest_3\n"
        "property float f_rest_4\nproperty float f_rest_5\n"
        "property float f_rest_6\nproperty float f_rest_7\n"
        "property float f_rest_8\n"
        "end_header\n"
        "0 0 0 0.1 0.2 0.3 1 2 3 4 5 6 7 8 9\n"));

    GaussianSplatProcessing processing;
    ASSERT_TRUE(processing.loadPly(file.fileName().toStdString()))
        << processing.lastError();
    ASSERT_EQ(processing.points().size(), 1U);

    const GaussianPoint& point = processing.points().front();
    for (std::size_t coefficient = 0; coefficient < 9; ++coefficient) {
        EXPECT_FLOAT_EQ(
            point.sh_rest[coefficient], static_cast<float>(coefficient + 1));
    }
    EXPECT_FLOAT_EQ(point.sh_rest[9], 0.0f);
    EXPECT_EQ(processing.metadata().sh_degree, 1);
}

TEST(GaussianProcessingTest, RejectsIncompleteScaleGroup)
{
    QTemporaryFile file;
    ASSERT_TRUE(writeText(file,
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float scale_0\nproperty float scale_1\n"
        "end_header\n"
        "0 0 0 1 1\n"));

    GaussianSplatProcessing processing;
    EXPECT_FALSE(processing.loadPly(file.fileName().toStdString()));
    EXPECT_NE(processing.lastError().find("incomplete scale"), std::string::npos);
}
