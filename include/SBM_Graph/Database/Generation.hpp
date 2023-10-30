#ifndef SBM_GRAPH_DATABASE_GENERATION_HPP
#define SBM_GRAPH_DATABASE_GENERATION_HPP
#include <QJsonObject>
#include <SBM_Graph/Graph.hpp>
#include <SBM_Graph/Community_Mappings.hpp>
namespace SBM_Graph
{
    void generate_SBM_to_db(const QJsonObject& p)
{
    auto [edge_lists, vertex_list] = generate_planted_SBM_edges(p["N_pop"].toInt(), p["N_communities"].toInt(), p["p_in"].toDouble(), p["p_out"].toDouble(), p["seed"].toInt());

    auto vcm = create_vcm(vertex_list);
    auto ccm = ccm_from_vcm(edge_lists, vcm);
    auto ecm = ecm_from_vcm(edge_lists, vcm, ccm);

    auto from_list = Edge_t::get_from(edge_lists);
    auto to_list = Edge_t::get_to(edge_lists);

    vcm_insert(p["p_out_id"].toInt(), p["graph_id"].toInt(), vcm);
    ecm_insert(p["p_out_id"].toInt(), p["graph_id"].toInt(), ecm);
    edge_insert("edgelists", p["p_out_id"].toInt(), p["graph_id"].toInt(), from_list, to_list);

    auto ccm_from = Weighted_Edge_t::get_from(ccm);
    auto ccm_to = Weighted_Edge_t::get_to(ccm);
    auto ccm_weight = Weighted_Edge_t::get_weights(ccm);
    ccm_insert(p["p_out_id"].toInt(), p["graph_id"].toInt(), ccm_from, ccm_to, ccm_weight);
}
}


#endif
