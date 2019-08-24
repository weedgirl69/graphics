using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace Gltf.Tests
{
    public class TestGltf
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public TestGltf(ITestOutputHelper testOutputHelper) =>
            _testOutputHelper = testOutputHelper;

        [Theory]
        [InlineData("%GLTF_SAMPLE_MODELS_DIR%/2.0")]
        private async Task TestModels(string directoryExpression)
        {
            var directory = _testOutputHelper.Log("directory={0}",
                Environment.ExpandEnvironmentVariables(directoryExpression));
            Assert.NotNull(directory);

            var gltfFilesEnumerator = Directory.EnumerateFiles(directory, "*.gltf",
                SearchOption.AllDirectories);

            await Task.WhenAll(from gltfPath in gltfFilesEnumerator
                select ValidateModel(gltfPath));
        }

        private async Task ValidateModel(string path)
        {
            _testOutputHelper.WriteLine($"path={path}");
            var model = await Model.FromPathAsync(path);
            Assert.NotEmpty(model.NodeTransforms);
            Assert.NotEmpty(model.AccessorsBytes);
            Assert.All(model.AccessorsBytes, action: bytes => { Assert.False(bytes.IsEmpty); });
            Assert.NotEmpty(model.MeshIndexToPrimitives);
        }
    }


    internal static class TestOutputHelperExtensions
    {
        public static T Log<T>(this ITestOutputHelper testOutputHelper, string formatString, T value)
        {
            testOutputHelper.WriteLine(format: formatString, value?.ToString());
            return value;
        }
    }
}