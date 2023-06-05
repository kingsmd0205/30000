#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#define MAX_FILENAME_LENGTH 256
#define MAX_FILE_COUNT 100
#define MAX_SUBDIRECTORIES 100
#define MAX_FILE_SIZE 1024 

// 定义了文件的数据结构
typedef struct {
    char name[MAX_FILENAME_LENGTH]; // 文件名
    int size; // 文件大小（单位：字节）
    int readonly; // 是否只读
    int writeonly; // 是否只写
    char data[MAX_FILE_SIZE]; // 文件数据
} File;

// 定义了目录的数据结构
typedef struct Directory {
    char name[MAX_FILENAME_LENGTH]; // 目录名
    int file_count; // 目录下包含的文件数目
    File files[MAX_FILE_COUNT]; // 存储目录下的所有文件
    int num_subdirectories; // 目录下包含的子目录数目
    struct Directory* parent; // 当前目录的父目录
    struct Directory* subdirectories[MAX_SUBDIRECTORIES]; // 存储目录下的所有子目录
} Directory;

Directory root_directory;  // 根目录



pthread_mutex_t mutex;
pthread_cond_t cond;


void close_file(Directory* directory, const char* filename) {
    for (int i = 0; i < directory->file_count; i++) {
        File* file = &(directory->files[i]);
        if (strcmp(file->name, filename) == 0) {
            printf("File closed: %s\n", filename);
            return;
        }
    }
    printf("File not found: %s\n", filename);
}
// 写入文件内容
void write_to_file(const string& filename, const string& content) {
    ofstream file(filename);
    if (file.is_open()) {
        file << content;
        file.close();
        cout << "内容已成功写入文件 " << filename << endl;
    } else {
        cout << "无法打开文件 " << filename << " 进行写入操作" << endl;
    }
}

// 从文件读取内容
void read_from_file(const string& filename) {
    ifstream file(filename);
    if (file.is_open()) {
        stringstream buffer;
        buffer << file.rdbuf();
        string content = buffer.str();
        cout << "文件 " << filename << " 的内容为：" << endl;
        cout << content << endl;
        file.close();
    } else {
        cout << "无法打开文件 " << filename << " 进行读取操作" << endl;
    }
}

// 导出文件
void export_file(const string& filename, const string& export_directory, const string& export_filename) {
    ifstream file(filename);
    if (file.is_open()) {
        string export_path = export_directory + "/" + export_filename;
        ofstream export_file(export_path);
        if (export_file.is_open()) {
            export_file << file.rdbuf();
            export_file.close();
            cout << "文件已成功导出为 " << export_path << endl;
        } else {
            cout << "无法打开导出文件 " << export_filename << " 进行写入操作" << endl;
        }
        file.close();
    } else {
        cout << "无法打开文件 " << filename << " 进行导出操作" << endl;
    }
}

// 显示目录和文件
void show_directory(const string& path) {
    cout << "目录 " << path << " 中的内容：" << endl;
    for (const string& item : file_system) {
        size_t pos = item.find_last_of('/');
        if (pos != string::npos) {
            string directory = item.substr(0, pos);
            if (directory == path) {
                cout << "[D] " << item.substr(pos + 1) << endl;
            }
        }
    }
    for (const string& item : file_system) {
        size_t pos = item.find_last_of('/');
        if (pos == string::npos) {
            cout << "[F] " << item << endl;
        }
    }
}
void lseek_file(Directory* directory, const char* filename, int offset) {
    for (int i = 0; i < directory->file_count; i++) {
        File* file = &(directory->files[i]);
        if (strcmp(file->name, filename) == 0) {
            if (offset < 0 || offset >= file->size) {
                printf("Invalid lseek operation\n");
                return;
            }
            printf("Lseek file: %s, Offset: %d\n", filename, offset);
            return;
        }
    }
    printf("File not found: %s\n", filename);
}

void rename_file(Directory* directory, const char* old_filename, const char* new_filename) {
    for (int i = 0; i < directory->file_count; i++) {
        File* file = &(directory->files[i]);
        if (strcmp(file->name, old_filename) == 0) {
            strcpy(file->name, new_filename);
            printf("File renamed: %s to %s\n", old_filename, new_filename);
            return;
        }
    }
    printf("File not found: %s\n", old_filename);
}

void import_file(Directory* directory, const char* filename) {
   
    for (int i = 0; i < directory->file_count; i++) {
        File* file = &(directory->files[i]);
        if (strcmp(file->name, filename) == 0) {
            printf("File already exists: %s\n", filename);
            return;
        }
    }

    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("File not found: %s\n", filename);
        return;
    }

   
    fseek(file, 0, SEEK_END);
    int file_size = ftell(file);
    fseek(file, 0, SEEK_SET);


    if (file_size > MAX_FILE_SIZE) {
        printf("File size exceeds maximum allowed size: %s\n", filename);
        fclose(file);
        return;
    }

    if (directory->file_count >= MAX_FILE_COUNT) {
        printf("Max file count exceeded in the directory. Cannot import file.\n");
        fclose(file);
        return;
    }


    File new_file;
    strcpy(new_file.name, filename);
    new_file.size = file_size;
    new_file.readonly = 0;
    new_file.writeonly = 1;

    fread(new_file.data, 1, file_size, file);

    fclose(file);


    directory->files[directory->file_count++] = new_file;

    printf("File imported: %s\n", filename);
}


void print_help() {
    printf("Available commands:\n");
    printf("cd <directory> - Change directory\n");
    printf("dir - List files in current directory\n");
    printf("mkdir <directory> - Create a new directory\n");
    printf("rmdir <directory> - Remove a directory\n");
    printf("create <filename> <size> <readonly> <writeonly> - Create a new file\n");
    printf("open <filename> - Open a file\n");
    printf("read <filename> <offset> <count> - Read from a file\n");
    printf("write <filename> <offset> <count> - Write to a file\n");
    printf("close <filename> - Close a file\n");
    printf("lseek <filename> <offset> - Set the file offset\n");
    printf("rename <old_filename> <new_filename> - Rename a file\n");
    printf("import <filename> - Import a file\n");
    printf("export <filename> - Export a file\n");
    printf("help - Show available commands\n");
    printf("time - Show current time\n");
    printf("ver - Show system version\n");
    printf("exit - Exit the program\n");
}

void print_time() {
    time_t current_time;
    time(&current_time);
    printf("Current time: %s", ctime(&current_time));
}


void print_version(Directory* current_directory) {
    printf("MyProgram v1.0\n");
    printf("Current Directory: %s\n", current_directory->name);
}


int main() {
// 初始化文件系统
    init_file_system();

    // 启动后台线程
    pthread_t background_thread; // 定义一个后台线程变量，用于表示线程ID
    pthread_mutex_init(&mutex, NULL); // 初始化互斥量，第二个参数为互斥量属性，一般置空即可
    pthread_cond_init(&cond, NULL); // 初始化条件变量，第二个参数同上
    pthread_create(&background_thread, NULL, background_thread_function, NULL); // 创建后台线程，传入线程函数的地址

    // 初始化用户输入参数
    char command[100];
    char arg1[MAX_FILENAME_LENGTH];
    char arg2[MAX_FILENAME_LENGTH];
    char arg3[MAX_FILENAME_LENGTH];
    char arg4[MAX_FILENAME_LENGTH];
    char arg5[MAX_FILENAME_LENGTH];

    Directory* current_directory = &root_directory; // 当前目录初始化为根目录

 while (1) {
        printf("Enter a command: ");
        fgets(command, sizeof(command), stdin); // 从用户输入的命令行中读取输入字符串
        sscanf(command, "%s", arg1); // 将用户输入字符流按格式化字符串转化为字符串数组

        if (strcmp(arg1, "cd") == 0) {
            sscanf(command, "%*s %s", arg2);
            change_directory(&current_directory, arg2);
        } else if (strcmp(arg1, "dir") == 0) {
            list_files(current_directory);
        } else if (strcmp(arg1, "mkdir") == 0) {
            sscanf(command, "%*s %s", arg2);
            make_directory(current_directory, arg2);
        } else if (strcmp(arg1, "rmdir") == 0) {
            sscanf(command, "%*s %s", arg2);
            remove_directory(current_directory, arg2);
        } else if (strcmp(arg1, "create") == 0) {
            sscanf(command, "%*s %s %s %s %s", arg2, arg3, arg4, arg5);
            int size = atoi(arg3);
            int readonly = atoi(arg4);
            int writeonly = atoi(arg5);
            create_file(current_directory, arg2, size, readonly, writeonly);
        } else if (strcmp(arg1, "open") == 0) {
            sscanf(command, "%*s %s", arg2);
            open_file(current_directory, arg2);
        } else if (strcmp(arg1, "read") == 0) {
            sscanf(command, "%*s %s %s %s", arg2, arg3, arg4);
            int offset = atoi(arg3);
            int count = atoi(arg4);
            read_file(current_directory, arg2, offset, count);
        } else if (strcmp(arg1, "write") == 0) {
            sscanf(command, "%*s %s %s %s", arg2, arg3, arg4);
            int offset = atoi(arg3);
            int count = atoi(arg4);
            write_file(current_directory, arg2, offset, count);
        } else if (strcmp(arg1, "close") == 0) {
            sscanf(command, "%*s %s", arg2);
            close_file(current_directory, arg2);
        } else if (strcmp(arg1, "lseek") == 0) {
            sscanf(command, "%*s %s %s", arg2, arg3);
            int offset = atoi(arg3);
            lseek_file(current_directory, arg2, offset);
        } else if (strcmp(arg1, "rename") == 0) {
            sscanf(command, "%*s %s %s", arg2, arg3);
            rename_file(current_directory, arg2, arg3);
        } else if (strcmp(arg1, "import") == 0) {
            sscanf(command, "%*s %s", arg2);
            import_file(current_directory, arg2);
        } else if (strcmp(arg1, "export") == 0) {
            sscanf(command, "%*s %s", arg2);
            export_file(current_directory, arg2);
        } else if (strcmp(arg1, "help") == 0) {
            print_help();
        } else if (strcmp(arg1, "time") == 0) {
            print_time();
        } else if (strcmp(arg1, "ver") == 0) {
            print_version(current_directory);
        } else if (strcmp(arg1, "exit") == 0) {
            break;
        } else {
            printf("Invalid command. Type 'help' to see available commands.\n");//未知命令的错误消息
        }
    }


    pthread_cancel(background_thread);  // 取消后台线程
    pthread_join(background_thread, NULL);  // 等待后台线程结束

    pthread_mutex_destroy(&mutex);  // 销毁互斥量
    pthread_cond_destroy(&cond);  // 销毁条件变量

    return 0;
}

