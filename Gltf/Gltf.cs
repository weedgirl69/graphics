using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Numerics;
using System.Text.Json;
using System.Threading.Tasks;

namespace Gltf
{
    public struct AffineTransform
    {
        public (Vector4, Vector4, Vector4) Columns;
    }

    public struct IndexData
    {
        public int ByteOffset;
        public int IndexSize;
    }

    public struct Primitive
    {
//      bounds: Bounds
        public int Count;
        public IndexData? IndexData;
        public int PositionsByteOffset;
        public int MaterialIndex;
        public int? NormalsByteOffset;
    }

    public class Model
    {
        private Model(ReadOnlyCollection<AffineTransform> nodeTransforms,
            ReadOnlyCollection<ReadOnlyMemory<byte>> accessorsBytes,
            ReadOnlyCollection<ReadOnlyCollection<Primitive>> meshIndexToPrimitives) =>
            (NodeTransforms, AccessorsBytes, MeshIndexToPrimitives) =
            (nodeTransforms, accessorsBytes, meshIndexToPrimitives);

        public ReadOnlyCollection<AffineTransform> NodeTransforms { get; }
        public ReadOnlyCollection<ReadOnlyMemory<byte>> AccessorsBytes { get; }

        public ReadOnlyCollection<ReadOnlyCollection<Primitive>> MeshIndexToPrimitives { get; }

        public static async Task<Model> FromPathAsync(string path)
        {
            await using var stream = File.OpenRead(path);
            using var jsonDocument = await JsonDocument.ParseAsync(stream);
            var gltfJson = jsonDocument.RootElement;
            var directoryName = Path.GetDirectoryName(path) ?? "";
            return new Model(nodeTransforms: GetNodeTransforms(gltfJson),
                accessorsBytes: await GetAccessorsBytes(gltfJson: gltfJson, directory: directoryName),
                meshIndexToPrimitives: GetMeshIndexToPrimitives(gltfJson)
            );
        }

        private static ReadOnlyCollection<ReadOnlyCollection<Primitive>> GetMeshIndexToPrimitives(JsonElement gltfJson)
        {
            var indicesMemory = new List<byte>();
            var indicesAccessorIndexToByteOffset = new Dictionary<int, int>();
            var accessorsJson = gltfJson.GetProperty("accessors");
            var meshIndexToPrimitives = new List<ReadOnlyCollection<Primitive>>();
            foreach (var meshJson in gltfJson.GetProperty("meshes").EnumerateArray())
            {
                var primitives = new List<Primitive>();
                foreach (var primitiveJson in meshJson.GetProperty("primitives").EnumerateArray())
                {
                    var primitive = new Primitive();
                    if (primitiveJson.TryGetProperty(propertyName: "indices", value: out var indicesAccessorIndexJson))
                    {
                        var indicesAccessorIndex = indicesAccessorIndexJson.GetInt32();

                        if (!indicesAccessorIndexToByteOffset.TryGetValue(key: indicesAccessorIndex,
                            value: out var indicesByteOffset))
                        {
                            indicesByteOffset = indicesMemory.Count;
                            indicesAccessorIndexToByteOffset.Add(key: indicesAccessorIndex, value: indicesByteOffset);

                            var indicesAccessorJson = accessorsJson[indicesAccessorIndex];
                            var componentType = indicesAccessorJson.GetProperty("componentType").GetInt32();
                            if (componentType == 5121)
                            {
                                
                            }
                        }
                    }

                    primitives.Add(primitive);
                }

                meshIndexToPrimitives.Add(primitives.AsReadOnly());
            }

            return meshIndexToPrimitives.AsReadOnly();
        }

        private static async Task<ReadOnlyCollection<ReadOnlyMemory<byte>>> GetAccessorsBytes(
            JsonElement gltfJson, string directory)
        {
            var buffersBytes = new List<Task<byte[]>>();
            foreach (var bufferJson in gltfJson.GetProperty("buffers").EnumerateArray())
                if (bufferJson.TryGetProperty(propertyName: "uri", value: out var uriJson))
                {
                    var uri = uriJson.GetString();
                    buffersBytes.Add(
                        uri.StartsWith("data:application/")
                            // ReSharper disable once StringIndexOfIsCultureSpecific.1
                            ? Task.FromResult(Convert.FromBase64String(uri.Substring(uri.IndexOf(",") + 1)))
                            : File.ReadAllBytesAsync(Path.Join(path1: directory, path2: uri)));
                }

            var bufferViewsJson = gltfJson.GetProperty("bufferViews");
            var accessorsBytes = new List<ReadOnlyMemory<byte>>();
            foreach (var accessorJson in gltfJson.GetProperty("accessors").EnumerateArray())
            {
                var count = accessorJson.GetProperty("count").GetInt32();
                var byteOffset = accessorJson.GetInt32Property("byteOffset") ?? 0;
                if (accessorJson.TryGetProperty(propertyName: "bufferView", value: out var bufferViewIndexJson))
                {
                    var bufferViewIndex = bufferViewIndexJson.GetInt32();
                    var bufferViewJson = bufferViewsJson[bufferViewIndex];
                    byteOffset += bufferViewJson.GetInt32Property("byteOffset") ?? 0;
                    var componentSize =
                        accessorJson.GetProperty("componentType").GetInt32() switch
                        {
                            5120 => 1,
                            5121 => 1,
                            5122 => 2,
                            5123 => 2,
                            5125 => 4,
                            5126 => 4,
                            var _ => throw new Exception()
                        };
                    var componentCount = accessorJson.GetProperty("type").GetString() switch
                    {
                        "SCALAR" => 1,
                        "VEC2" => 2,
                        "VEC3" => 3,
                        "VEC4" => 4,
                        "MAT2" => 4,
                        "MAT3" => 9,
                        "MAT4" => 16,
                        var _ => throw new Exception()
                    };
                    var naturalStride = componentSize * componentCount;
                    var byteStride = bufferViewJson.GetInt32Property("byteStride") ?? naturalStride;
                    var byteCount = count * byteStride;
                    var bufferBytes = await buffersBytes[bufferViewJson.GetProperty("buffer").GetInt32()];
                    if (byteStride == naturalStride)
                    {
                        accessorsBytes.Add(
                            bufferBytes.AsSpan(start: byteOffset, length: byteCount).ToArray());
                    }
                    else
                    {
                        var accessorBytes = new byte[byteCount];
                        for (var i = 0; i < count; ++i)
                            bufferBytes.AsSpan(start: byteOffset + i * byteStride, length: naturalStride)
                                .CopyTo(accessorBytes.AsSpan(i * naturalStride));
                        accessorsBytes.Add(accessorBytes);
                    }
                }
                else
                {
                    accessorsBytes.Add(new byte[count]);
                }
            }

            return accessorsBytes.AsReadOnly();
        }

        private static ReadOnlyCollection<AffineTransform> GetNodeTransforms(JsonElement gltfJson)
        {
            var nodeTransforms = new List<AffineTransform>();
            foreach (var nodeJson in gltfJson.GetProperty("nodes").EnumerateArray())
                if (nodeJson.TryGetProperty(propertyName: "matrix", value: out var matrixJson))
                {
                    nodeTransforms.Add(new AffineTransform
                    {
                        Columns = (
                            new Vector4(
                                x: (float) matrixJson[0].GetDouble(),
                                y: (float) matrixJson[4].GetDouble(),
                                z: (float) matrixJson[8].GetDouble(),
                                w: (float) matrixJson[12].GetDouble()),
                            new Vector4(
                                x: (float) matrixJson[1].GetDouble(),
                                y: (float) matrixJson[5].GetDouble(),
                                z: (float) matrixJson[9].GetDouble(),
                                w: (float) matrixJson[13].GetDouble()),
                            new Vector4(
                                x: (float) matrixJson[2].GetDouble(),
                                y: (float) matrixJson[6].GetDouble(),
                                z: (float) matrixJson[10].GetDouble(),
                                w: (float) matrixJson[14].GetDouble()))
                    });
                }
                else
                {
                    var rotation = nodeJson.TryGetProperty(propertyName: "rotation", value: out var rotationJson)
                        ? new Vector4(x: rotationJson[0].GetSingle(), y: rotationJson[1].GetSingle(),
                            z: rotationJson[2].GetSingle(), w: rotationJson[3].GetSingle())
                        : Vector4.Zero;

                    var scale = nodeJson.TryGetProperty(propertyName: "scale", value: out var scaleJson)
                        ? new Vector3(x: scaleJson[0].GetSingle(), y: scaleJson[1].GetSingle(),
                            z: scaleJson[2].GetSingle())
                        : Vector3.One;


                    var translation = nodeJson.TryGetProperty(propertyName: "scale", value: out var translationJson)
                        ? new Vector3(x: translationJson[0].GetSingle(), y: translationJson[1].GetSingle(),
                            z: translationJson[2].GetSingle())
                        : Vector3.Zero;

                    var xx = rotation.X * rotation.X;
                    var xy = rotation.X * rotation.Y;
                    var xz = rotation.X * rotation.Z;
                    var xw = rotation.X * rotation.W;
                    var yy = rotation.Y * rotation.Y;
                    var yz = rotation.Y * rotation.Z;
                    var yw = rotation.Y * rotation.W;
                    var zz = rotation.Z * rotation.Z;
                    var zw = rotation.Z * rotation.W;

                    var m00 = 1 - 2 * (yy + zz);
                    var m01 = 2 * (xy - zw);
                    var m02 = 2 * (xz + yw);

                    var m10 = 2 * (xy + zw);
                    var m11 = 1 - 2 * (xx + zz);
                    var m12 = 2 * (yz - xw);

                    var m20 = 2 * (xz - yw);
                    var m21 = 2 * (yz + xw);
                    var m22 = 1 - 2 * (xx + yy);

                    nodeTransforms.Add(new AffineTransform
                    {
                        Columns = (
                            new Vector4(x: scale.X * m00,
                                y: scale.Y * m01,
                                z: scale.Z * m02,
                                w: translation.X),
                            new Vector4(x: scale.X * m10,
                                y: scale.Y * m11,
                                z: scale.Z * m12,
                                w: translation.Y),
                            new Vector4(x: scale.X * m20,
                                y: scale.Y * m21,
                                z: scale.Z * m22,
                                w: translation.Z)
                        )
                    });
                }

            return nodeTransforms.AsReadOnly();
        }
    }

    internal static class Extensions
    {
        public static int? GetInt32Property(this JsonElement jsonElement, string propertyName) =>
            jsonElement.TryGetProperty(propertyName: propertyName, value: out var propertyElement)
                ? propertyElement.GetInt32()
                : (int?) null;
    }
}