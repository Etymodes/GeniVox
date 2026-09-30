using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

namespace GeniVox
{
    [ComImport]
    [Guid("00021401-0000-0000-C000-000000000046")]
    internal class ShellLink { }

    [StructLayout(LayoutKind.Sequential)]
    internal struct PropertyKey
    {
        public Guid FormatId;
        public uint PropertyId;
    }

    [StructLayout(LayoutKind.Explicit, Size = 24)]
    internal struct PropVariant
    {
        [FieldOffset(0)] public ushort Type;
        [FieldOffset(8)] public IntPtr Value;
    }

    [ComImport]
    [Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IPropertyStore
    {
        void GetCount(out uint count);
        void GetAt(uint index, out PropertyKey key);
        void GetValue(ref PropertyKey key, out PropVariant value);
        void SetValue(ref PropertyKey key, ref PropVariant value);
        void Commit();
    }

    public static class ShortcutAppId
    {
        private static PropertyKey AppIdKey = new PropertyKey {
            FormatId = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"),
            PropertyId = 5
        };

        public static void Set(string path, string appId)
        {
            object link = new ShellLink();
            try
            {
                IPersistFile file = (IPersistFile)link;
                file.Load(path, 2); // STGM_READWRITE
                IPropertyStore properties = (IPropertyStore)link;
                PropVariant value = new PropVariant {
                    Type = 31, // VT_LPWSTR
                    Value = Marshal.StringToCoTaskMemUni(appId)
                };
                try
                {
                    properties.SetValue(ref AppIdKey, ref value);
                    properties.Commit();
                    file.Save(path, true);
                }
                finally
                {
                    Marshal.FreeCoTaskMem(value.Value);
                }
            }
            finally
            {
                Marshal.ReleaseComObject(link);
            }
        }

        public static string Get(string path)
        {
            object link = new ShellLink();
            try
            {
                ((IPersistFile)link).Load(path, 0);
                PropVariant value;
                ((IPropertyStore)link).GetValue(ref AppIdKey, out value);
                try
                {
                    return value.Type == 31 ? Marshal.PtrToStringUni(value.Value) : null;
                }
                finally
                {
                    PropVariantClear(ref value);
                }
            }
            finally
            {
                Marshal.ReleaseComObject(link);
            }
        }

        [DllImport("ole32.dll")]
        private static extern int PropVariantClear(ref PropVariant value);
    }
}
