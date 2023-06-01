#ifndef TMVA_BATCHGENERATOR
#define TMVA_BATCHGENERATOR

#include <iostream>
#include <vector>
#include <thread>
#include <memory>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "TMVA/RChunkLoader.hxx"
#include "TMVA/RBatchLoader.hxx"
#include "TMVA/Tools.h"
#include "TRandom3.h"

namespace TMVA {
namespace Experimental {

template<typename... Args>
class RBatchGenerator
{
private:
    TMVA::RandomGenerator<TRandom3> rng;

    std::vector<std::string> cols, filters;
    size_t num_columns, chunk_size, max_chunks, batch_size, current_row=0, entries;

    std::string file_name, tree_name;
    
    std::unique_ptr<TMVA::Experimental::RBatchLoader> batch_loader;

    std::unique_ptr<std::thread> loading_thread;
    bool initialized = false;

    bool EoF = false, use_whole_file = true;
    double validation_split;

    std::unique_ptr<TMVA::Experimental::RTensor<float>> chunk_tensor;

    std::vector<std::vector<size_t>> training_idxs;
    std::vector<std::vector<size_t>> validation_idxs;

    std::vector<size_t> vec_sizes;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Functions
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // Load chunk_size rows of the given RDataFrame into a RTensor.
    // After, the chunk of data is split into batches of data.
    void LoadChunk(size_t current_chunk)
    {
        TMVA::Experimental::RChunkLoader<Args...> func((*chunk_tensor), vec_sizes);

        // Create DataFrame
        long long start_l = current_row;
        long long end_l = start_l + chunk_size;
        ROOT::RDF::Experimental::RDatasetSpec x_spec = ROOT::RDF::Experimental::RDatasetSpec().AddSample({"",tree_name,
                                                file_name}).WithGlobalRange( {start_l, std::numeric_limits<Long64_t>::max()});
        ROOT::RDataFrame x_rdf(x_spec);

        size_t progressed_events, passed_events;

        // add filters if given
        if (filters.size() > 0) {
            auto x_filter = x_rdf.Filter(filters[0], "F1");

            for (auto i = 1; i < filters.size(); i++) {
                auto name = "F" + std::to_string(i);
                x_filter = x_filter.Filter(filters[i], name);
            }

            // add range
            auto x_ranged = x_filter.Range(chunk_size);
            auto myReport = x_ranged.Report();

            // load data
            x_ranged.Foreach(func, cols);

            // get the loading info
            progressed_events = myReport.begin()->GetAll();
            passed_events = (myReport.end()-1)->GetPass();
        }
        
        // no filters given
        else {
            // add range
            auto x_ranged = x_rdf.Range(chunk_size);
            auto myCount = x_ranged.Count();

            // load data
            x_ranged.Foreach(func, cols);

            // get loading info
            progressed_events = myCount.GetValue();
            passed_events = myCount.GetValue();
        }
        
        // std::cout << "RBatchGenerator::init => tensor: " << chunk_tensor << std::endl;

        current_row += progressed_events;

        // Create batches for the current_chunk.
        // First get the correct idices to use, then turn them into batches
        // Validation batches only have to be made in the first epoch
        if (training_idxs.size() > current_chunk) {
            batch_loader->CreateTrainingBatches(*chunk_tensor, training_idxs[current_chunk]);
        }
        else {
            createIdxs(current_chunk, progressed_events);
            batch_loader->CreateTrainingBatches(*chunk_tensor, training_idxs[current_chunk]);
            batch_loader->CreateValidationBatches(*chunk_tensor, validation_idxs[current_chunk]);
        }
    }
    void createIdxs(size_t current_chunk, size_t progressed_events)
    {
        std::vector<size_t> row_order = std::vector<size_t>(progressed_events);

        std::iota(row_order.begin(), row_order.end(), 0);

        std::shuffle(row_order.begin(), row_order.end(),rng);

        size_t num_validation = progressed_events * validation_split;

        std::vector<size_t> valid_idx({row_order.begin(), row_order.begin() + num_validation});
        std::vector<size_t> train_idx({row_order.begin() + num_validation, row_order.end()});

        training_idxs.push_back(train_idx);
        validation_idxs.push_back(valid_idx);
    }

public:

    RBatchGenerator(std::string _file_name, std::string _tree_name, std::vector<std::string> _cols, 
                   std::vector<std::string> _filters, size_t _chunk_size, size_t _batch_size, std::vector<size_t> _vec_sizes = {}, 
                   double _validation_split=0.0, size_t _max_chunks = 0, size_t _num_columns = 0):
    file_name(_file_name), tree_name(_tree_name), cols(_cols), filters(_filters), num_columns(_num_columns), 
    chunk_size(_chunk_size), batch_size(_batch_size), vec_sizes(_vec_sizes), validation_split(_validation_split), max_chunks(_max_chunks) 
    {
        
        if (max_chunks > 0) {use_whole_file = false;};

        if (num_columns == 0){
            num_columns = cols.size();
        }

        // get the number of entries in the dataframe
        TFile* f = TFile::Open(file_name.c_str());
        TTree* t = f->Get<TTree>(tree_name.c_str());
        entries = t->GetEntries();

        std::cout << "RBatchGenerator => found " << entries << " entries in file." << std::endl;

        batch_loader = std::make_unique<TMVA::Experimental::RBatchLoader>(batch_size, num_columns);

        chunk_tensor = std::unique_ptr<TMVA::Experimental::RTensor<float>>(new TMVA::Experimental::RTensor<float>({chunk_size, num_columns}));

        rng = TMVA::RandomGenerator<TRandom3>(0);
    }

    ~RBatchGenerator()
    {
        StopLoading();
    } 

    void StopLoading()
    {
        if (loading_thread) {
            loading_thread->join();
            loading_thread = nullptr;
        }
    }

    void init()
    {
        std::cout << "Cpp::RBatchGenerator => init" << std::endl;

        current_row = 0;
        batch_loader->Activate();
        loading_thread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunks, this);
    }

    // Returns the next batch of data if available. 
    // Returns empty RTensor otherwise.
    TMVA::Experimental::RTensor<float>* GetTrainBatch()
    {   

        // Get next batch if available
        if (batch_loader->HasTrainData()) {
            TMVA::Experimental::RTensor<float>* batch = batch_loader->GetTrainBatch();
            return batch;
        }

        // return empty batch if all events have been used
        return new TMVA::Experimental::RTensor<float>({0,0});
    }

    // Returns the next batch of data if available. 
    // Returns empty RTensor otherwise.
    TMVA::Experimental::RTensor<float>* GetValidationBatch()
    {   
        // Get next batch if available
        if (batch_loader->HasValidationData()) {
            return batch_loader->GetValidationBatch();
        }
        
        // return empty batch if all events have been used
        return new TMVA::Experimental::RTensor<float>({0,0});
    }

    bool HasTrainData()
    {
        if (!batch_loader->HasTrainData() && EoF) {
            return false;
        }

        return true;
    }

    bool HasValidationData()
    {
        if (!batch_loader->HasValidationData()) {
            return false;
        }

        return true;
    }
    
    
    void LoadChunks()
    {
        EoF = false;
        
        // Load chunks untill the end of the file is reached. 
        // Stop loading if a maximum number of chunks is provided
        for (size_t i = 0; ((i < max_chunks) || use_whole_file); i++) {
            LoadChunk(i);
            if (current_row >= entries) {
                break;
            }
        }    

        batch_loader->DeActivate();
        EoF = true;
    }

    void start_validation()
    {
        batch_loader->start_validation();
    }

};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHGENERATOR