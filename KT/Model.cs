namespace KT {
  using System;
  using System.Collections.ObjectModel;
  using System.Numerics;

  public partial struct Model {
    private Model(ReadOnlyCollection<AffineTransform> nodeTransforms,
                  ReadOnlyCollection<ReadOnlyMemory<byte>> accessorsBytes,
                  ReadOnlyMemory<byte> indicesBytes,
                  ReadOnlyCollection<
                    ReadOnlyCollection<Primitive>> meshIndexToPrimitives) =>
      (this.NodeTransforms, this.AccessorsBytes, this.IndicesBytes, this.MeshIndexToPrimitives) =
      (nodeTransforms, accessorsBytes, indicesBytes, meshIndexToPrimitives);

    public ReadOnlyCollection<AffineTransform> NodeTransforms { get; }
    public ReadOnlyCollection<ReadOnlyMemory<byte>> AccessorsBytes { get; }
    public ReadOnlyMemory<byte> IndicesBytes { get; }

    public ReadOnlyCollection<
      ReadOnlyCollection<Primitive>> MeshIndexToPrimitives { get; }
  }

  public struct AffineTransform {
    public (Vector4, Vector4, Vector4) Columns;
  }

  public struct IndexData {
    public int ByteOffset;
    public int IndexSize;
  }

  public struct Primitive {
//      bounds: Bounds
    public int Count;
    public IndexData? IndexData;
    public int PositionsByteOffset;
    public int MaterialIndex;
    public int? NormalsByteOffset;
  }
}